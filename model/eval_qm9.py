import sys, os
import glob
import warnings
warnings.filterwarnings('ignore')
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#from torch_geometric.data import DataLoader
import numpy as np

from sklearn.model_selection import train_test_split

from load_data import load_torch_from_dicts, load_split_fn

from FragGraph import FragGraph
from train import *


print_= print
def print(*args, **kwargs):
  kwargs['flush']=True
  print_(*args, **kwargs)

def graph_summary(graph):
  print(graph)
  print('nodes:\n  {}'.format(graph.nodes))
  print('  size:',graph.nodes.size(dim=-1))
  print('edges:\n  {}'.format(graph.edges))
  print('  size:',graph.edges.size(dim=-1))
  print('edge_indexs:\n  {}'.format(graph.edge_index))
  print('  size:',graph.edge_index.size(dim=-1))
  print('globals:\n  {}'.format(graph.globals))
  print('  size:',graph.globals.size(dim=-1))
  print('batch:\n  {}'.format(graph.batch))
  print('ptr:\n  {}'.format(graph.ptr))
  print('y:\n  {}'.format(graph.y))

def xyz2torch(fns, y_dict):
  if type(fns) != list:
    fns = [fns]
  data_lst = list()
  for fn in fns:
    x, pos, mol = xyz2mol(fn)
    data = mol2torch(mol, y=y_dict[fn.split('/')[-1]])
    if data is not None:
      data_lst.append(data)
  return data_lst

def load_targets(targ_fn, which_idx=-1):
  if type(targ_fn) != list:
    targ_fn = [ targ_fn ]
  targ_dict = dict()
  for targ_fn_ in targ_fn:
    inlines = [x.split(' ') for x in list(filter(None, open(targ_fn_, 'r').read().split('\n')))]
    targ_dict['header_'+targ_fn_] = inlines[0][which_idx] 
    for line in inlines[1:]:
      targ_dict[line[0].split('/')[-1].split('.')[0]] = float(line[which_idx])
  return targ_dict

if __name__=="__main__":

  '''
   Options:
     load_pretrained_model 
       requires model_path defined
     load_split
       requires split_path
       or list of molecule labels and bool if in hold out set
     mode 
       0 - eval
       1 - train
  '''
  load_pretrained_model = True
  model_path = 'FragGraph_fc_256.pth'
  load_split = True
  split_path = 'data/gdb9_hold_out.txt'
  mode = 0 

  # Change path to attr graphs if needed
  path_to_gdb9_attr = 'data/gdb9_fragment_embedded_graphs'

  # load data
  y_dict = load_targets('data/qm9_properties.txt')
  fns = sorted(list(glob.glob(f'{path_to_gdb9_attr}/*.npy')))[:2]

  if load_split:
    splits = load_split_fn(split_path)
  else:
    splits = None

  data_lst_tr, data_lst_te, label_lst_tr, label_lst_te = load_torch_from_dicts(fns, y_dict, splits=splits)
  tr_full_len = len(data_lst_tr)
  te_full_len = len(data_lst_te)
  print('Loaded {} train and {} test'.format(tr_full_len, te_full_len))
  if te_full_len == 0:
    print('te_full_len == ',te_full_len)
    data_lst_te = data_lst_tr
    label_lst_te = label_lst_tr
  X_tr = data_lst_tr
  X_te = data_lst_te
   
  print('Creating DataLoaders')
  X_tr_batches = [b for b in DataLoader(X_tr, batch_size=32, shuffle=False)]
  X_te_batches = [b for b in DataLoader(X_te, batch_size=32, shuffle=False)]

  # load or initialize model
  if load_pretrained_model:
    model = load_model(model_path)
  else:
    model = FragGraph(batches[0])

  # train model
  if mode == 1:
    model = train(model, batches, num_iters=10, load_model=False, save_model=False) 

  # eval model
  print('Evaluating QM9 data set   MSE   MAE   RMSE  (kcal/mol)')
  y_pred_tr = predict(model, X_tr_batches, 'gdb9 train')
  y_pred_te = predict(model, X_te_batches, 'gdb9 test')
  
