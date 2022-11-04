import sys, os
import numpy as np
import torch 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_data(files):
  graphs = list()
  np_load_old = np.load
  np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
  for i, fn in enumerate(files):
    print('Loading file {:02d} / {:02d} : {}'.format(i+1,len(files),fn))
    graphs.extend(np.load(fn).tolist())
  np.load = np_load_old
  return graphs

def load_pop(pop_file):
  np_load_old = np.load
  np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
  pop_dict = np.load(pop_file).tolist()[0]
  np.load = np_load_old
  print('Loaded {} pops from {}'.format(len(pop_dict.keys()), pop_file))
  return pop_dict

def load_split_fn(fns):
  if type(fns) != list:
    fns = [ fns ]
  splits = []
  for fn in fns:
    try:
      splits.extend([x.split(" ") for x in list(filter(None,open(fn, "r").read().split("\n")))])
    except:
      print('could not find',fn)
      continue
  split_dict=dict()
  for x in splits:
    split_dict[x[0].split('/')[-1].split('.')[0]]=x[1]
  return split_dict

def graph2torch(graph, y=None, fc=True, pop_nodes=None):
  nodes = torch.tensor(np.asarray(graph['nodes']), dtype=torch.float32)
  if not fc:
    new_edges = list()
    new_senders = list()
    new_receivers = list()
    for idx, e in enumerate(graph['edges']):
      if float(e[0]) > 0.:
        new_edges.append(e)
        new_senders.append(graph['senders'][idx])
        new_receivers.append(graph['receivers'][idx])
    graph['edges'] = new_edges
    graph['senders'] = new_senders
    graph['receivers'] = new_receivers
  if len(graph['edges']) > 0: # mol has bonds
    edges = torch.tensor(np.asarray(graph['edges']), dtype=torch.float32)
    edge_index = torch.tensor(np.asarray([graph['senders'],graph['receivers']]), dtype=torch.long)
  else:
    return None
  g = torch.tensor(np.asarray([graph['globals']]), dtype=torch.float32)
  y = torch.tensor(np.array([[y]]), dtype=torch.float32)
  atom_types = list()
  for node in graph['nodes']:
    atom_types.append(node[:6])  
  atom_types = torch.tensor(np.asarray(atom_types), dtype=torch.float32)
  if pop_nodes is not None:
    pop_nodes_ = torch.tensor(np.asarray(pop_nodes), dtype=torch.float32)
    data = Data(nodes=nodes, edge_index=edge_index, edges=edges, globals=g, y=y, pop_nodes=pop_nodes_, atom_types=atom_types)
  else:
    data = Data(nodes=nodes, edge_index=edge_index, edges=edges, globals=g, y=y, atom_types=atom_types)
  return data

def load_torch_from_dicts(files, y_dict, splits=None, fc=True, override_IP=False, pop=None, override_global=False, global_fn='data/esp_wb97xd_large_no_collapse_w_frag_vec.txt', simple_graphs=False, use_zero_pop=False):
  if type(files) != list:
    files = [ files ]
  if type(splits) != dict:
    splits = {}
  graphs = load_data(files)
  graphs_lst_tr, graphs_lst_te = list(), list()
  pop_err = True
  labels_lst_tr, labels_lst_te = list(), list()
  global_d = dict()
  if override_global:
    inlines = [x.split(' ') for x in list(filter(None, open(global_fn, 'r').read().split('\n')))]
    for x in inlines:
      global_d[x[0]] = x[1:]
  for g in graphs:
    label = g['label'].split('/')[-1].split('.')[0]
    if pop is not None:
      sys.exit()
      label_ = label.replace('_neu','').replace('_cat','')
      if label_ in pop:
        pop_nodes = list()
        pop_err = False
        pop_lst = pop[label_]
        node_idx = 0
        for pop_idx, pop_d in enumerate(pop_lst):
          if pop_d[0] == 1:
            continue
          else:
            if g['nodes'][node_idx][[6, 7, 8, 9, 16, 17].index(pop_d[0])] == 1.0:
              pop_v = [pop_d[-1]]
              pop_nodes.append(pop_v)
              node_idx += 1
            else:
              print('ERROR pop {} doesnt match'.format(pop_idx))
              pop_err = True
      elif use_zero_pop:
        pop_nodes = list()
        pop_err = False
        for node in g['nodes']:
          pop_nodes.append([0.])
      else:
        print('ERROR: {} not in pop_dict'.format(label))
    if override_global:
      g['globals'] = [float(x_) for x_ in global_d[label]]

    if override_IP:
      if 'qm7b' in label:
        label = label.replace('qm7b_','qm7b_IP_')
    if simple_graphs:
      sys.exit()
      n_lst = list()
      for node in g['nodes']:
        n_lst.append(node[0:6])
      g['nodes'] = n_lst

    if label in y_dict:
      pop_err = True
      if pop is not None:
        if len(pop_nodes) != len(g['nodes']):
          pop_err = True
        else:
          pop_err = False
      if not pop_err:
        torch_data = graph2torch(g, y_dict[label], fc=fc, pop_nodes=pop_nodes)  
      else:
        torch_data = graph2torch(g, y_dict[label], fc=fc)
      if torch_data is not None:
        if label not in splits:
          new_label_ = label.replace('_cat','').replace('_neu','')
          if new_label_ in splits:
            label = new_label_
        if label in splits:
          if splits[label] == 'True':
            graphs_lst_te.append(torch_data)
            labels_lst_te.append(label)
            continue
        graphs_lst_tr.append(torch_data)
        labels_lst_tr.append(label)
    else:
      print('ERROR: {} not in y_dict'.format(label)) 
  return graphs_lst_tr, graphs_lst_te, labels_lst_tr, labels_lst_te

if __name__=="__main__":
  files = ['gdb9/attr_graphs/gdb9_skeleton_graphs_0a_attr.npy','gdb9/attr_graphs/gdb9_skeleton_graphs_0b_attr.npy']

  graphs = load_data(files)
  torchs = [graph2torch(graph, 1.0) for graph in graphs]
  print(len(graphs), len(torchs))
  for g, t_g in zip(graphs, torchs):
    if t_g is not None:
      print(g)
      for k, v in g.items():
        print(k, np.asarray(v).shape)
      print(t_g)
      break
