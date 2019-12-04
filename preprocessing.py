import csv
import pickle
import os
import numpy as np
import scipy.sparse as sp
from collections import OrderedDict
# functions
def load_variable(fn,):
    op_interaction_list=open(fn,'rb')
    return pickle.load(op_interaction_list)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def Norm(raw_info):
    X=np.array(list(raw_info.values()))
    std1 = np.nanstd(X, axis=0)
    feat_filt = std1!=0
    X = np.ascontiguousarray(X[:,feat_filt])
    means1 = np.mean(X, axis=0)
    X = np.tanh((X-means1)/std1[feat_filt])
    means2 = np.mean(X, axis=0)
    std2 = np.std(X, axis=0)
    X = (X-means2)/std2
    X[:,std2==0]=0
    feat=X
    return feat

def get_feat(raw_info):
    feat=Norm(raw_info)
    nonzero_feat, num_feat = feat.shape
    feat=sp.csr_matrix(feat)
    feat = sparse_to_tuple(feat.tocoo())
    return feat,nonzero_feat,num_feat

def get_raw_input_file(file_name,sep='\t',Encoding='UTF-8',if_np=True):
    input_tmp=[i.strip('\n').strip('\r').split(sep) for i in open(file_name,encoding=Encoding).readlines()]
    if if_np:
        return np.array(input_tmp)
    else:
        return input_tmp

def extending_set(multi_list):
    List=[]
    for i in multi_list:
        List.extend(i)
    List=sorted(list(set(List)))
    return List

def extending(multi_list):
    List=[]
    for i in multi_list:
        List.extend(i)
    return List

def sset(List,Sorted=True):
    if Sorted:
        new_list=sorted(list(set(List)))
    else:
        new_list=list(set(List))
    return new_list