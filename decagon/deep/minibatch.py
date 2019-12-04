from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp

from ..utility import preprocessing

np.random.seed(123)
class EdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.
    assoc -- numpy array with target edges
    placeholders -- tensorflow placeholders object
    batch_size -- size of the minibatches
    """
    def __init__(self, adj_mats, feat, edge_types,nsz_fold,tra_f,tra_edges_t,tra_edges_f,test_edges_t,test_edges_f,val_edges_t,val_edges_f,\
                 edge_type2idx,idx2edge_type,labels_for_reg,iter_len,batch_size, val_test_size):
        self.adj_mats = adj_mats
        self.feat = feat
        self.edge_types = edge_types
        self.batch_size = batch_size
        self.val_test_size = val_test_size
        self.num_edge_types = sum(self.edge_types.values())

        self.iter = 0
        self.freebatch_edge_types= list(range(self.num_edge_types))
        self.batch_num = [0]*self.num_edge_types
        self.batch_num_f = [0]*self.num_edge_types
        self.current_edge_type_idx = 0
        self.nsz_fold = nsz_fold
        self.edge_type2idx = edge_type2idx
        self.idx2edge_type = idx2edge_type
        self.tra_f=tra_f
        self.neg_batch_index = {i:list(range(len(self.tra_f[i]))) for i in range(self.num_edge_types)}
        self.test_edges_t = test_edges_t
        self.test_edges_f = test_edges_f
        self.val_edges_t = val_edges_t
        self.val_edges_f = val_edges_f
        self.tra_edges_t = tra_edges_t
        self.tra_edges_f = tra_edges_f
        self.labels_for_reg = labels_for_reg
        self.iter_len = iter_len

        self.train_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        self.val_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        self.test_edges = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        self.test_edges_false = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        self.val_edges_false = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}

        # Function to build test and val sets with val_test_size positive links
        self.adj_train = {edge_type: [None]*n for edge_type, n in self.edge_types.items()}
        for i, j in self.edge_types:
            for k in range(self.edge_types[i,j]):
                print("\nMinibatch edge type:", "(%d, %d, %d)" % (i, j, k))
                if (i,j)==(1, 1):
                    self.mask_test_edges_dd((i, j), k)
                else:
                    self.mask_test_edges((i, j), k)
                print("Train edges=", "%04d" % len(self.train_edges[i,j][k]),\
                      "Val edges=", "%04d" % len(self.val_edges[i,j][k]),\
                      "Test edges=", "%04d" % len(self.test_edges[i,j][k]))

    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        if adj.shape[0] == adj.shape[1]:
            adj_ = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj_.sum(1))
            degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
            adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        else:
            rowsum = np.array(adj.sum(1))
            colsum = np.array(adj.sum(0))
            rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
            coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
            adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
        return preprocessing.sparse_to_tuple(adj_normalized)

    def _ismember(self, a, b):
        a = np.array(a)
        b = np.array(b)
        rows_close = np.all(a - b == 0, axis=1)
        return np.any(rows_close)

    def mask_test_edges_dd(self, edge_type, type_idx):
        edges_all, _, _ = preprocessing.sparse_to_tuple(self.adj_mats[edge_type][type_idx])

        test_edges = self.test_edges_t[type_idx]
        val_edges = self.val_edges_t[type_idx]
        train_edges = np.array(self.tra_edges_t[type_idx])
        test_edges_false =self.test_edges_f[type_idx]
        val_edges_false =self.val_edges_f[type_idx]

        # Re-build adj matrices
        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix(
            (data, (train_edges[:, 0], train_edges[:, 1])),
            shape=self.adj_mats[edge_type][type_idx].shape)
        self.adj_train[edge_type][type_idx] = self.preprocess_graph(adj_train)

        self.train_edges[edge_type][type_idx] = train_edges
        self.val_edges[edge_type][type_idx] = np.array(val_edges)
        self.val_edges_false[edge_type][type_idx] = np.array(val_edges_false)
        self.test_edges[edge_type][type_idx] = np.array(test_edges)
        self.test_edges_false[edge_type][type_idx] = np.array(test_edges_false)

    def mask_test_edges(self, edge_type, type_idx):
        edges_all, _, _ = preprocessing.sparse_to_tuple(self.adj_mats[edge_type][type_idx])
        num_test = max(5, int(np.floor(edges_all.shape[0] * self.val_test_size)))
        num_val = max(5, int(np.floor(edges_all.shape[0] * self.val_test_size)))

        all_edge_idx = list(range(edges_all.shape[0]))
        np.random.shuffle(all_edge_idx)

        val_edge_idx = all_edge_idx[:num_val]
        val_edges = edges_all[val_edge_idx]

        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges_all[test_edge_idx]

        train_edges = np.delete(edges_all, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        test_edges_false = []
        while len(test_edges_false) < len(test_edges)*1:
            if len(test_edges_false) % 3000 == 0:
                print("Constructing test edges=", "%04d/%04d" % (len(test_edges_false), len(test_edges)))
            idx_i = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[0])
            idx_j = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if self._ismember([idx_i, idx_j], test_edges_false):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            if len(val_edges_false) % 3000 == 0:
                print("Constructing val edges=", "%04d/%04d" % (len(val_edges_false), len(val_edges)))
            idx_i = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[0])
            idx_j = np.random.randint(0, self.adj_mats[edge_type][type_idx].shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if val_edges_false:
                if self._ismember([idx_i, idx_j], val_edges_false):
                    continue
            val_edges_false.append([idx_i, idx_j])

        # Re-build adj matrices
        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix(
            (data, (train_edges[:, 0], train_edges[:, 1])),
            shape=self.adj_mats[edge_type][type_idx].shape)
        self.adj_train[edge_type][type_idx] = self.preprocess_graph(adj_train)

        self.train_edges[edge_type][type_idx] = train_edges
        self.val_edges[edge_type][type_idx] = val_edges
        self.val_edges_false[edge_type][type_idx] = np.array(val_edges_false)
        self.test_edges[edge_type][type_idx] = test_edges
        self.test_edges_false[edge_type][type_idx] = np.array(test_edges_false)

    def end(self):
        finished = len(self.freebatch_edge_types) == 0
        return finished

    def update_feed_dict(self, feed_dict, dropout, placeholders):
        # construct feed dictionary
        feed_dict.update({
            placeholders['adj_mats_%d,%d,%d' % (i,j,k)]: self.adj_train[i,j][k]
            for i, j in self.edge_types for k in range(self.edge_types[i,j])})
        feed_dict.update({placeholders['feat_%d' % i]: self.feat[i] for i, _ in self.edge_types})
        feed_dict.update({placeholders['dropout']: dropout})
        return feed_dict

    def batch_feed_dict(self, batch_edges,batch_neg, batch_pos_s,batch_neg_s,batch_edge_type, placeholders):
        feed_dict = dict()
        feed_dict.update({placeholders['batch']: batch_edges}) #self.batch_feed_dict(batch_edges, self.current_edge_type_idx, placeholders)
        feed_dict.update({placeholders['batch_neg']: batch_neg})
        feed_dict.update({placeholders['batch_pos_s']: batch_pos_s})
        feed_dict.update({placeholders['batch_neg_s']: batch_neg_s})
        feed_dict.update({placeholders['batch_edge_type_idx']: batch_edge_type})
        feed_dict.update({placeholders['batch_row_edge_type']: self.idx2edge_type[batch_edge_type][0]})
        feed_dict.update({placeholders['batch_col_edge_type']: self.idx2edge_type[batch_edge_type][1]})
        return feed_dict

    def next_minibatch_feed_dict(self, placeholders):
        """Select a random edge type and a batch of edges of the same type"""
        while True:
            if len(self.freebatch_edge_types) > 0:
                self.current_edge_type_idx = np.random.choice(self.freebatch_edge_types)
            else:
                self.current_edge_type_idx = self.edge_type2idx[0, 0, 0]
                self.iter = 0 #

            i, j, k = self.idx2edge_type[self.current_edge_type_idx]
            Rest=len(self.train_edges[i,j][k])-(self.batch_num[self.current_edge_type_idx]*self.batch_size)
            if Rest>=self.batch_size:
                Enough=True
                break
            else:
                if self.iter % self.iter_len <=3:
                    self.batch_num[self.current_edge_type_idx] = 0
                else:
                    Enough=False
                    break

        if Enough:
            self.iter += 1
            start = self.batch_num[self.current_edge_type_idx] * self.batch_size
            self.batch_num[self.current_edge_type_idx] += 1
            batch_edges = self.train_edges[i,j][k][start: start + self.batch_size]
        else:
            self.iter += 1
            start = self.batch_num[self.current_edge_type_idx] * self.batch_size
            self.batch_num[self.current_edge_type_idx] += 1
            batch_edges1 = self.train_edges[i,j][k][start:]
            batch_edges2=self.train_edges[i,j][k][:self.batch_size-Rest]
            batch_edges = np.vstack((batch_edges1,batch_edges2))
            self.freebatch_edge_types.remove(self.current_edge_type_idx)
        batch_pos=[]
        #nsz_fold=30
        nsz=self.batch_size*self.nsz_fold
        neg_ind_start= self.batch_num_f[self.current_edge_type_idx]*nsz
        neg_ind_end= self.batch_num_f[self.current_edge_type_idx]*nsz+nsz
        batch_neg1=self.tra_f[self.current_edge_type_idx][neg_ind_start:neg_ind_end]
        self.batch_num_f[self.current_edge_type_idx]+=1
        if len(batch_neg1)==nsz:
            batch_neg=batch_neg1

        else:
            vacancy=nsz-len(batch_neg1)
            batch_neg2=self.tra_f[self.current_edge_type_idx][:vacancy]
            self.batch_num_f[self.current_edge_type_idx]=0
            batch_neg = np.vstack((batch_neg1,batch_neg2))

        for i,j in batch_edges:
            for num in range(self.nsz_fold):
                batch_pos.append([i,j])
        self.batch_neg=np.array(batch_neg)
        self.batch_edges=np.array(batch_edges)

        if self.current_edge_type_idx>3:
            self.batch_pos_s=np.array([self.labels_for_reg[self.current_edge_type_idx-4][tuple(i)] for i in self.batch_edges])
            self.batch_neg_s=np.array([self.labels_for_reg[self.current_edge_type_idx-4][tuple(i)] for i in self.batch_neg])
        else:
            self.batch_pos_s=np.array([1. for i in self.batch_edges])
            self.batch_neg_s=np.array([0. for i in self.batch_neg])
        return self.batch_feed_dict(self.batch_edges, self.batch_neg,self.batch_pos_s, self.batch_neg_s,self.current_edge_type_idx, placeholders)

    def num_training_batches(self, edge_type, type_idx):
        return len(self.train_edges[edge_type][type_idx]) // self.batch_size + 1

    def val_feed_dict(self, edge_type, type_idx, placeholders, size=None):
        edge_list = self.val_edges[edge_type][type_idx]
        if size is None:
            return self.batch_feed_dict(edge_list, edge_type, placeholders)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges, edge_type, placeholders)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        for edge_type in self.edge_types:
            for k in range(self.edge_types[edge_type]):
                self.train_edges[edge_type][k] = np.random.permutation(self.train_edges[edge_type][k])
                self.batch_num[self.edge_type2idx[edge_type[0], edge_type[1], k]] = 0
                self.batch_num_f[self.edge_type2idx[edge_type[0], edge_type[1], k]] = 0
        for i in range(self.num_edge_types):
            self.tra_f[i]= np.random.permutation(self.tra_f[i])

        self.current_edge_type_idx = 0
        self.freebatch_edge_types = list(range(self.num_edge_types))
        self.freebatch_edge_types.remove(self.edge_type2idx[0, 0, 0])
        self.freebatch_edge_types.remove(self.edge_type2idx[0, 0, 1])
        self.freebatch_edge_types.remove(self.edge_type2idx[0, 1, 0])
        self.freebatch_edge_types.remove(self.edge_type2idx[1, 0, 0])
        self.iter = 0