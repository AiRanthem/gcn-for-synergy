from __future__ import division
from __future__ import print_function

from collections import OrderedDict, Counter
import time
import os
import numpy as np
import scipy.sparse as sp

from operator import itemgetter
from sklearn import metrics
from scipy import stats
from preprocessing import *
import random
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

np.random.seed()

from decagon.deep.minibatch import EdgeMinibatchIterator
from decagon.deep.optimizer import DecagonOptimizer
from decagon.deep.model import DecagonModel

# 我添加的部分
best_EPOCH = 2

CUDA_VISIBLE_DEVICES = '1'
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

Model_Parameters = OrderedDict()
Model_Parameters['Negative sample size'] = 30
Model_Parameters['Neg_sample_weights'] = 1
Model_Parameters['Norm'] = 1
Model_Parameters['threshold'] = 30
Model_Parameters['Sep_set'] = [['4', '3', '0'], ['1'], ['2']]  # [[train sets],[validation set],[test set]]
Model_Parameters['ecoder'] = 'dedicom'
Model_Parameters['Initial learning rate'] = 0.001
Model_Parameters['Max epoch'] = best_EPOCH  # best_EPOCH
Model_Parameters['hidden layer'] = [2048, 1024]
# Model_Parameters['hidden layer']=[64, 32]
Model_Parameters['Weight for L2 loss on embedding matrix'] = 0
Model_Parameters['Dropout rate'] = 0.2
Model_Parameters['l2'] = 1
Model_Parameters['Max margin parameter in hinge loss'] = 0.1
Model_Parameters['minibatch size'] = 20
Model_Parameters['Bias term'] = True
Model_Parameters['val_test_size'] = 0.05
Model_Parameters['LOSS'] = 'Cross-entropy'

PRINT_PROGRESS_EVERY = 20

# Data loading and preprocessing
cel_adj = load_variable('dataset/cel_adj')
cel_drug_adj = load_variable('dataset/cel_drug_adj')
drug_cel_adj = load_variable('dataset/drug_cel_adj')

raw_labels = get_raw_input_file('dataset/drug combinations.csv', sep=',')[1:]
labels = OrderedDict()
for i in raw_labels:
    line, da, db, cel, ss, fold = i
    if line in labels:
        labels[line] = [(labels[line][0] + float(ss)) / 2, fold]
    else:
        labels[line] = [float(ss), fold]

all_drugs = extending_set([i.split('_')[:2] for i in labels.keys()])
n_drugs = len(all_drugs)
cellLine = sset([i.split('_')[2] for i in labels.keys()])
n_cel = len(cellLine)
idx_drugs = {i[1]: i[0] for i in enumerate(all_drugs)}
idx_cellLine = {i[1]: i[0] for i in enumerate(cellLine)}

labels_for_reg = {j: {} for j in range(len(idx_cellLine) * 2)}
drug_drug_adj_list = [np.zeros((n_drugs, n_drugs)) for i in cellLine]
folds = {i: [] for i in ['4', '2', '3', '1', '0']}
for entry, content in labels.items():
    da, db, cel = entry.split('_')
    synergy_score = content[0]
    labels_for_reg[idx_cellLine[cel]][(idx_drugs[da], idx_drugs[db])] = float(
        synergy_score > Model_Parameters['threshold'])
    labels_for_reg[idx_cellLine[cel]][(idx_drugs[db], idx_drugs[da])] = float(
        synergy_score > Model_Parameters['threshold'])
    labels_for_reg[idx_cellLine[cel] + n_cel][(idx_drugs[da], idx_drugs[db])] = float(
        synergy_score > Model_Parameters['threshold'])
    labels_for_reg[idx_cellLine[cel] + n_cel][(idx_drugs[db], idx_drugs[da])] = float(
        synergy_score > Model_Parameters['threshold'])
    if synergy_score >= Model_Parameters['threshold']:
        drug_drug_adj_list[idx_cellLine[cel]][idx_drugs[da]][idx_drugs[db]] = 1
        drug_drug_adj_list[idx_cellLine[cel]][idx_drugs[db]][idx_drugs[da]] = 1
    folds[content[1]].append(entry)

drug_drug_adj = [sp.csr_matrix(i) for i in drug_drug_adj_list]


def get_set(fold_list, threshold):
    aim_fold = extending([folds[i] for i in fold_list])
    edges_t = {i: [] for i in list(range(n_cel * 2))}
    edges_f = {i: [] for i in list(range(n_cel * 2))}
    for i in aim_fold:
        d1, d2, cel = i.split('_')
        cel_ind = idx_cellLine[cel]
        d1_idx, d2_idx = idx_drugs[d1], idx_drugs[d2]
        if labels[i][0] >= threshold:
            edges_t[cel_ind].append([d1_idx, d2_idx])
            edges_t[cel_ind].append([d2_idx, d1_idx])
            edges_t[cel_ind + n_cel].append([d1_idx, d2_idx])
            edges_t[cel_ind + n_cel].append([d2_idx, d1_idx])
        else:
            edges_f[cel_ind].append([d1_idx, d2_idx])
            edges_f[cel_ind].append([d2_idx, d1_idx])
            edges_f[cel_ind + n_cel].append([d1_idx, d2_idx])
            edges_f[cel_ind + n_cel].append([d2_idx, d1_idx])
    return edges_t, edges_f


# Combination of training set and validation set
tra_edges_t, tra_edges_f = get_set(Model_Parameters['Sep_set'][0] + Model_Parameters['Sep_set'][1],
                                   Model_Parameters['threshold'])
val_edges_t, val_edges_f = get_set(Model_Parameters['Sep_set'][1], Model_Parameters['threshold'])  # suspending
test_edges_t, test_edges_f = get_set(Model_Parameters['Sep_set'][2], Model_Parameters['threshold'])

cel_info = load_variable('dataset/cel_info')
cel_feat, cel_nonzero_feat, cel_num_feat = get_feat(cel_info)

drug_info = load_variable('dataset/drug_info')
drug_feat, drug_nonzero_feat, drug_num_feat = get_feat(drug_info)

cel_degrees = np.array(cel_adj.sum(axis=0)).squeeze()
drug_degrees_list = [np.array([len(np.flatnonzero(row)) for row in drug_adj.toarray()]) for drug_adj in drug_drug_adj]

adj_mats_orig = {
    (0, 0): [cel_adj, cel_adj.transpose(copy=True)],
    (0, 1): [cel_drug_adj],
    (1, 0): [drug_cel_adj],
    (1, 1): drug_drug_adj + [x.transpose(copy=True) for x in drug_drug_adj],
}

degrees = {
    0: [cel_degrees, cel_degrees],
    1: drug_degrees_list + drug_degrees_list,
}

num_feat = {
    0: cel_num_feat,
    1: drug_num_feat,
}
nonzero_feat = {
    0: cel_nonzero_feat,
    1: drug_nonzero_feat,
}
feat = {
    0: cel_feat,
    1: drug_feat,
}

edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_type2decoder = {
    (0, 0): 'bilinear',
    (0, 1): 'bilinear',
    (1, 0): 'bilinear',
    (1, 1): 'dedicom',
}

edge_types = {k: len(v) for k, v in adj_mats_orig.items()}
num_edge_types = sum(edge_types.values())

r = 0
edge_type2idx = {}
idx2edge_type = {}
for i, j in edge_types:
    for k in range(edge_types[i, j]):
        edge_type2idx[i, j, k] = r
        idx2edge_type[r] = i, j, k
        r += 1

tra_f_no_et = OrderedDict()
for ind, et in idx2edge_type.items():
    if ind < 4:
        mat = adj_mats_orig[et[:2]][et[2]]
        tra_f_no_et[ind] = np.argwhere(1 - mat.toarray()).tolist()
    else:
        tra_f_no_et[ind] = [list(i) for i in tra_edges_f[ind - 4]]

idx2name = {}
for k, v in idx2edge_type.items():
    if k >= 4:
        idx2name[k] = cellLine[v[2] % n_cel]
    else:
        idx2name[k] = idx2edge_type[i]

###########################################################
# Settings and placeholders
###########################################################
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', Model_Parameters['Initial learning rate'], 'Initial learning rate.')
flags.DEFINE_integer('Max_epoch', Model_Parameters['Max epoch'], 'Max epoch.')
flags.DEFINE_float('weight_decay', Model_Parameters['Weight for L2 loss on embedding matrix'],
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', Model_Parameters['Dropout rate'], 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('l2', Model_Parameters['l2'], 'l2')
flags.DEFINE_float('max_margin', Model_Parameters['Max margin parameter in hinge loss'],
                   'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', Model_Parameters['minibatch size'], 'minibatch size.')
flags.DEFINE_boolean('bias', Model_Parameters['Bias term'], 'Bias term.')


def construct_placeholders(edge_types):
    placeholders = {
        'batch': tf.placeholder(tf.int32, name='batch'),
        'batch_neg': tf.placeholder(tf.int32, name='batch_neg'),
        'batch_pos_s': tf.placeholder(tf.float32, name='batch_pos_s'),
        'batch_neg_s': tf.placeholder(tf.float32, name='batch_neg_s'),
        'batch_edge_type_idx': tf.placeholder(tf.int32, shape=(), name='batch_edge_type_idx'),
        'batch_row_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_row_edge_type'),
        'batch_col_edge_type': tf.placeholder(tf.int32, shape=(), name='batch_col_edge_type'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i, j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders


print("*****Defining placeholders*****")
placeholders = construct_placeholders(edge_types)

###########################################################
# Create minibatch iterator, model and optimizer
###########################################################
print("\n***Create minibatch iterator***")
minibatch = EdgeMinibatchIterator(
    adj_mats=adj_mats_orig,
    feat=feat,
    edge_types=edge_types,
    batch_size=FLAGS.batch_size,
    val_test_size=Model_Parameters['val_test_size'],
    nsz_fold=Model_Parameters['Negative sample size'],
    tra_f=tra_f_no_et,
    test_edges_f=test_edges_f,
    test_edges_t=test_edges_t,
    val_edges_f=val_edges_f,
    val_edges_t=val_edges_t,
    tra_edges_t=tra_edges_t,
    tra_edges_f=tra_edges_f,
    labels_for_reg=labels_for_reg,
    edge_type2idx=edge_type2idx,
    idx2edge_type=idx2edge_type,
    iter_len=sum(edge_types.values()),
)

print("\n***Create model***")
model = DecagonModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    edge_types=edge_types,
    hidden_layer=Model_Parameters['hidden layer'],
    l2=FLAGS.l2,
    decoders=edge_type2decoder,
)

print("\n***Create optimizer***")
with tf.name_scope('optimizer'):
    opt = DecagonOptimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_types=edge_types,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin,
        neg_sample_weights=Model_Parameters['Neg_sample_weights'],
        LOSS=Model_Parameters['LOSS'],
    )
print("\n***Initialize session***")

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
feed_dict = {}


###########################################################
# Train model
###########################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_accuracy_scores(edges_pos, edges_neg, edge_type):
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['batch_edge_type_idx']: edge_type2idx[edge_type]})  # current_edge_type_idx
    feed_dict.update({placeholders['batch_row_edge_type']: edge_type[0]})
    feed_dict.update({placeholders['batch_col_edge_type']: edge_type[1]})

    cost, rec = sess.run([opt.cost, opt.predictions], feed_dict=feed_dict)

    predicted_dic = OrderedDict()
    if edge_type2idx[edge_type] > 3:
        preds_score = []
        labels_all = []
        for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
            ave = (rec[u, v] + rec[v, u]) / 2
            predicted_dic[u, v] = ave
            predicted_dic[v, u] = ave
            labels_all.append(1)
            preds_score.append(ave)
        for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
            ave = (rec[u, v] + rec[v, u]) / 2
            predicted_dic[u, v] = ave
            predicted_dic[v, u] = ave
            labels_all.append(0)
            preds_score.append(ave)

        preds_score = np.array(preds_score)
        pred = np.array([0] * len(preds_score))
        pred[sigmoid(preds_score) > 0.5] = 1

        roc_sc = metrics.roc_auc_score(labels_all, preds_score)
        aupr_sc = metrics.average_precision_score(labels_all, preds_score)
        # mse=metrics.mean_squared_error(synergy_score, preds_score)
        # pearson=stats.pearsonr(synergy_score, preds_score)[0]
        bacc = metrics.balanced_accuracy_score(labels_all, pred)
        acc = metrics.accuracy_score(labels_all, pred)
        prec = metrics.precision_score(labels_all, pred)
    else:
        preds_score = []
        labels_all = []
        for u, v in edges_pos[edge_type[:2]][edge_type[2]]:
            predicted_dic[u, v] = rec[u, v]
            predicted_dic[v, u] = rec[u, v]
            labels_all.append(1)
            preds_score.append(rec[u, v])
        for u, v in edges_neg[edge_type[:2]][edge_type[2]]:
            predicted_dic[u, v] = rec[u, v]
            predicted_dic[v, u] = rec[u, v]
            labels_all.append(0)
            preds_score.append(rec[u, v])
        roc_sc = metrics.roc_auc_score(labels_all, preds_score)
        aupr_sc = metrics.average_precision_score(labels_all, preds_score)
        prec, bacc, acc = 0, 0, 0
    return roc_sc, aupr_sc, bacc, acc, prec, rec, cost, predicted_dic


print("\n***Train model***")
train_loss_eve = OrderedDict()
train_loss = OrderedDict()

for raw_epoch in range(FLAGS.Max_epoch):
    epoch = raw_epoch
    print("\nEpoch:", "%04d" % (epoch + 1), "time: ", time.ctime()[:-5])
    train_loss_eve[epoch] = []
    minibatch.shuffle()
    itr = 0
    save_epoch = False
    while not minibatch.end():
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)
        outs = sess.run([opt.opt_op, opt.cost, opt.batch_edge_type_idx], feed_dict=feed_dict)
        train_cost = outs[1]
        batch_edge_type = int(outs[2])
        if minibatch.current_edge_type_idx > 3:
            train_loss_eve[epoch].append(train_cost)
        if itr % PRINT_PROGRESS_EVERY == 0:
            print("Edge:",
                  "%-3s%-14s  tra_loss=%f " % (batch_edge_type, idx2name[minibatch.current_edge_type_idx], train_cost))
        itr += 1
    train_loss[epoch] = sum(train_loss_eve[epoch]) / len(train_loss_eve[epoch])
print("\n***Training finished!***")

output_accuracy_scores = OrderedDict()
output_accuracy_scores['minibatch.idx2edge_type'] = ['Edge type', 'ROC AUC', \
                                                     'PR AUC', 'MSE', 'Pearson', 'Bacc', 'Acc', 'Count']
cellLine1 = cellLine + [i + '*' for i in cellLine]
rec = OrderedDict()
predicted_dic = OrderedDict()
for et in range(num_edge_types):
    if idx2edge_type[et][:2] == (1, 1):
        edge_Type = cellLine1[et - 4]
        count = len(test_edges_t[et - 4])
        rec_log = True
    else:
        edge_Type = idx2edge_type[et][:2]
        count = ' '
        rec_log = False
    try:
        roc_score, auprc_score, test_bacc, test_acc, test_rec, test_loss, et_predicted_dic = get_accuracy_scores(
            minibatch.test_edges, minibatch.test_edges_false, idx2edge_type[et])
        if rec_log:
            rec[edge_Type] = test_rec
        predicted_dic[et] = et_predicted_dic
        print("Edge type:", "%04d" % et, "ROC AUC", "{:.4f}".format(roc_score), \
              "PR AUC", "{:.4f}".format(auprc_score), \
              "MSE", "{:.4f}".format(test_mse), \
              "Pearson", "{:.4f}".format(test_pearson))
        output_accuracy_scores[str(minibatch.idx2edge_type[et])] = [edge_Type, roc_score, auprc_score, test_mse,
                                                                    test_pearson, \
                                                                    test_bacc, test_acc, count]
    except:
        print('ERROR:', et)
        output_accuracy_scores[str(minibatch.idx2edge_type[et])] = [edge_Type, 'ERROR']
