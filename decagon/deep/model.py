from collections import defaultdict

import tensorflow as tf

from .layers import GraphConvolutionMulti, GraphConvolutionSparseMulti, \
    DistMultDecoder, InnerProductDecoder, DEDICOMDecoder, BilinearDecoder

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class DecagonModel(Model):
    def __init__(self, placeholders, num_feat, nonzero_feat, edge_types, hidden_layer,l2,decoders, **kwargs):
        super(DecagonModel, self).__init__(**kwargs)
        self.edge_types = edge_types
        self.num_edge_types = sum(self.edge_types.values())
        self.num_obj_types = max([i for i, _ in self.edge_types]) + 1
        self.decoders = decoders
        self.inputs = {i: placeholders['feat_%d' % i] for i, _ in self.edge_types}
        self.input_dim = num_feat
        self.hidden_layer = hidden_layer
        self.nonzero_feat = nonzero_feat
        self.placeholders = placeholders
        self.dropout = placeholders['dropout']
        self.l2 = l2
        self.adj_mats = {et: [
            placeholders['adj_mats_%d,%d,%d' % (et[0], et[1], k)] for k in range(n)]
            for et, n in self.edge_types.items()}
        self.build()

    def _build(self):
        self.hidden1 = defaultdict(list)
        for i, j in self.edge_types:
            self.hidden1[i].append(GraphConvolutionSparseMulti(
                input_dim=self.input_dim, output_dim=self.hidden_layer[0],
                edge_type=(i,j), num_types=self.edge_types[i,j],
                adj_mats=self.adj_mats, nonzero_feat=self.nonzero_feat,
                act=tf.nn.relu, l2=self.l2,dropout=self.dropout,
                logging=self.logging)(self.inputs[j]))

        for i, hid1 in self.hidden1.items():
            self.hidden1[i] = tf.nn.relu(tf.add_n(hid1))
        self.hidden_last=self.hidden1

        for layer_n in range(1,len(self.hidden_layer)-1):
            self.hidden_new = defaultdict(list)
            for i, j in self.edge_types:
                self.hidden_new[i].append(GraphConvolutionMulti(
                    input_dim=self.hidden_layer[layer_n-1], output_dim=self.hidden_layer[layer_n],
                    edge_type=(i,j), num_types=self.edge_types[i,j],
                    adj_mats=self.adj_mats,
                    act=tf.nn.relu, l2=self.l2,dropout=self.dropout,
                    logging=self.logging)(self.hidden_last[j]))
            for i, hid_n in self.hidden_new.items():
                self.hidden_new[i] = tf.nn.relu(tf.add_n(hid_n))
            self.hidden_last=self.hidden_new

        self.embeddings_reltyp = defaultdict(list)
        for i, j in self.edge_types:
            self.embeddings_reltyp[i].append(GraphConvolutionMulti(
                input_dim=self.hidden_layer[-2], output_dim=self.hidden_layer[-1],
                edge_type=(i,j), num_types=self.edge_types[i,j],
                adj_mats=self.adj_mats, l2=self.l2,act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.hidden_last[j]))

        self.embeddings = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp.items():
            self.embeddings[i] = tf.add_n(embeds)

        self.edge_type2decoder = {}
        for i, j in self.edge_types:
            decoder = self.decoders[i, j]
            if decoder == 'innerproduct':
                self.edge_type2decoder[i, j] = InnerProductDecoder(
                    input_dim=self.hidden_layer[-1], logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'distmult':
                self.edge_type2decoder[i, j] = DistMultDecoder(
                    input_dim=self.hidden_layer[-1], logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'bilinear':
                self.edge_type2decoder[i, j] = BilinearDecoder(
                    input_dim=self.hidden_layer[-1], logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)
            elif decoder == 'dedicom':
                self.edge_type2decoder[i, j] = DEDICOMDecoder(
                    input_dim=self.hidden_layer[-1], logging=self.logging,
                    edge_type=(i, j), num_types=self.edge_types[i, j],
                    act=lambda x: x, dropout=self.dropout)

            else:
                raise ValueError('Unknown decoder type')

        self.latent_inters = []
        self.latent_varies = []
        for edge_type in self.edge_types:
            decoder = self.decoders[edge_type]
            for k in range(self.edge_types[edge_type]):
                if decoder == 'innerproduct':
                    glb = tf.eye(self.hidden_layer[-1], self.hidden_layer[-1])
                    loc = tf.eye(self.hidden_layer[-1], self.hidden_layer[-1])
                elif decoder == 'distmult':
                    glb = tf.diag(self.edge_type2decoder[edge_type].vars['relation_%d' % k])
                    loc = tf.eye(self.hidden_layer[-1], self.hidden_layer[-1])
                elif decoder == 'bilinear':
                    glb = self.edge_type2decoder[edge_type].vars['relation_%d' % k]
                    loc = tf.eye(self.hidden_layer[-1], self.hidden_layer[-1])
                elif decoder == 'dedicom':
                    glb = self.edge_type2decoder[edge_type].vars['global_interaction']
                    loc = tf.diag(self.edge_type2decoder[edge_type].vars['local_variation_%d' % k])
                else:
                    raise ValueError('Unknown decoder type')

                self.latent_inters.append(glb)
                self.latent_varies.append(loc)
