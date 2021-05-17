import sys, os
from pprint import pprint
from rdkit import Chem
from rdkit.Chem import AllChem
from xyz2mol import xyz2mol as x2m
import smi2attr
import numpy as np
import math

class FragGraph:
  def __init__(self, name):
    self.name = name
    self.n, self.e, self.g = [], [], []
    self.senders, self.receivers = [], []
    self.check_graph()

  def check_graph(self):
    self.n_nodes = len(self.n)
    self.n_edges = len(self.e)
    if self.n_nodes == 0:
      self.valid=[False, 'FragGraph has no nodes']
    elif self.n_edges == len(self.senders) and self.n_edges == len(self.receivers):
      self.valid=[True, 'FragGraph is valid']
    else:
      self.valid=[False, 'FragGraph edges do not match']
    return

  def obj2dict(self):
    return {'label':self.name,
            'nodes':self.n,
            'edges':self.e,
            'globals':self.g,
            'senders':self.senders,
            'receivers':self.receivers}

  def addnode(self, node):
    self.n.append(node)
    return

  def addedge(self, i, j, edge):
    self.e.append(edge)
    self.senders.append(i)
    self.receivers.append(j)
    return

  def rmnode(self, node):
    if node < len(self.n):
      new_nodes=list()
      for idx, node_ in enumerate(self.n):
        if idx != node:
          new_nodes.append(node_)
      new_edges = list()
      new_senders, new_receivers = list(), list()
      for idx in range(len(self.e)):
        sender = self.senders[idx]
        receiver = self.receivers[idx]
        if sender != node and receiver != node:
          new_edges.append(self.e[idx])
          if sender > node:
            sender-=1
          if receiver > node:
            receiver-=1
          new_senders.append(sender)
          new_receivers.append(receiver)
      self.n = new_nodes.copy()
      self.e = new_edges.copy()
      self.senders = new_senders.copy()
      self.receivers = new_receivers.copy()
    return

def mol2fraggraph(mol,name,atoms=None,xyz=None):
  mol = Chem.AddHs(mol)
  Chem.Kekulize(mol, clearAromaticFlags=True)
  graph = FragGraph(name)
  for atom in mol.GetAtoms():
    graph.addnode([atom.GetIdx(),atom.GetSymbol()])
  for bond in mol.GetBonds():
    atom1=bond.GetBeginAtomIdx()
    atom2=bond.GetEndAtomIdx()
    edge_vec = [bond.GetBondTypeAsDouble()]
    if xyz is not None:
      d=get_dist(xyz[atom1],xyz[atom2])
    else:
      d=None
    edge_vec.extend([d,graph.n[atom1][1],graph.n[atom2][1]])
    graph.addedge(atom1,atom2,edge_vec)
    graph.addedge(atom2,atom1,edge_vec)
  return graph

def xyz2mol(fn_xyz):
  atoms, charge, xyz_coordinates = x2m.read_xyz_file(fn_xyz)
  mol = x2m.xyz2mol(atoms, xyz_coordinates, charge, embed_chiral=True)
  mol = Chem.AddHs(mol)
  Chem.Kekulize(mol, clearAromaticFlags=True)
  return atoms, xyz_coordinates, mol


def smi2mol(smi):
  mol = Chem.MolFromSmiles(smi)
  mol = Chem.AddHs(mol)
  Chem.Kekulize(mol, clearAromaticFlags=True)
  return mol

def fraginc2smi(inc,mol):
  smi = Chem.MolToSmiles(mol)
  mw = Chem.RWMol(mol)
  numatoms = mw.GetNumAtoms()
  total_deg = [atom.GetTotalValence() for atom in mw.GetAtoms()]
  for i in range(numatoms):
    idx = numatoms-1-i
    if idx not in inc:
      mw.RemoveAtom(idx)
  numatoms = mw.GetNumAtoms()
  mw = Chem.RWMol(Chem.AddHs(mw))
  for idx, val in enumerate(total_deg):
    if idx in inc:
      idx2 = sorted(list(set(inc))).index(idx)
      atom = mw.GetAtomWithIdx(idx2)
      if atom.GetAtomicNum() != 1:
        if atom.GetTotalValence() != val:
          for _ in range(val-atom.GetTotalValence()):
            idx_h = mw.AddAtom(Chem.Atom(1))
            mw.AddBond(idx2,idx_h,Chem.BondType.SINGLE)
  idx_rings = list()
  for r in Chem.GetSymmSSSR(mw):
    for x in r:
      if x not in idx_rings:
        idx_rings.append(x)
  for idx, atom in enumerate(mw.GetAtoms()):
    if idx not in idx_rings:
      atom.SetIsAromatic(False)
  if len(Chem.GetSymmSSSR(mw)) < 1:
    try:
      Chem.Kekulize(mw, clearAromaticFlags=True)
      smi = Chem.MolToSmiles(mw,kekuleSmiles=True,canonical=True)
    except:
      print('Cannot kekulize mw')
      smi = Chem.MolToSmiles(mw)

  #smi = Chem.MolToSmiles(mw,kekuleSmiles=True,canonical=True)
  else:
    smi = Chem.MolToSmiles(mw)
  mol = Chem.RemoveHs(Chem.MolFromSmiles(smi))
  smi = Chem.MolToSmiles(mol)
  return smi

def collect_neighbors(node, graph, incl_h=False):
  for x in node.copy():
    for i in range(len(graph.e)):
      sender = graph.senders[i]
      receiver = graph.receivers[i]
      if x == sender:
        if receiver not in node and graph.n[receiver][1] != 'H':
          node.append(receiver)
      elif x == receiver:
        if sender not in node and graph.n[sender][1] != 'H':
          node.append(sender)
  return node

def cbh_n(graph, rung=2):
  rung_ = 0
  nodes_ls = [[x[0]] if x[1] != 'H' else [] for x in graph.n]
  while rung_ < rung:
    for idx, node in enumerate(nodes_ls):
      nodes_ls[idx] = collect_neighbors(node, graph)
    rung_+=2
  return nodes_ls

def atom2label(label):
  atoms=['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']
  if label in atoms:
    return atoms.index(label)+1
  else:
    return label

def node_vec(node):
  smi=node[2][-1]
  attr = smi2attr.smi2attr(smi)
  vec, model = smi2attr.smi2mol2vec(smi)
  return [atom2label(node[1])]+attr+vec

def gaussian_expansion(d, n_centers=20, sigma=0.5, min_d=0.0, max_d=4.0, centers=[None]):
  if None in centers:
    centers = np.linspace(min_d, max_d, n_centers)
  if d is None:
    return [ 0. ] * n_centers , centers
  else:
    return [ math.exp(-(d - x)**2 / sigma**2) for x in centers ], centers

def get_dist(x1, x2):
  return round(((x1[0]-x2[0])**2+(x1[1]-x2[1])**2+(x1[2]-x2[2])**2)**0.5,8)

def bond_vec(edge):
  '''
    edge = [ bond_order, bond_length, atom_1, atom_2 ]
  '''
  bond_types = [1.0, 1.5, 2.0, 3.0]
  atom_types = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 31, 32, 33, 34, 35]
  one_hot_bond = [[edge[0]].count(x) for x in bond_types]
  one_hot_atom = [[edge[2]].count(x) for x in atom_types]+[[edge[3]].count(x) for x in atom_types]
  bond_exp = gaussian_expansion(edge[1])[0]
  return edge[0:2]+[atom2label(x) for x in edge[2:]]+one_hot_bond+one_hot_atom+bond_exp

def global_vec(smi):
  attr = smi2attr.smi2attr(smi)
  vec, model = smi2attr.smi2mol2vec(smi)
  return attr+vec

def xyz2fraggraph(fn, attr=False):
  atoms, xyz_coordinates, mol = xyz2mol(fn)
  f = mol2fraggraph(mol, fn, xyz=xyz_coordinates)
  fraginc_ls = cbh_n(f)
  for idx, incl in enumerate(fraginc_ls):
    if len(incl) > 0:
      frag_smi = fraginc2smi(incl, mol)
      f.n[idx].append([incl, frag_smi])
  h_atoms = [idx for idx, node in enumerate(f.n) if node[1] == 'H']
  for idx in sorted(h_atoms, reverse=True):
    f.rmnode(idx)
  if attr:
    for idx, node in enumerate(f.n):
      f.n[idx]=node_vec(node)
    for idx, edge in enumerate(f.e):
      f.e[idx]=bond_vec(edge)
    f.g=global_vec(Chem.MolToSmiles(mol))
    f_attr_dict=f.obj2dict()
  return f.obj2dict()

def smi2fraggraph(smi, attr=False):
  mol = smi2mol(smi)
  f = mol2fraggraph(mol, smi)
  fraginc_ls = cbh_n(f)
  for idx, incl in enumerate(fraginc_ls):
    if len(incl) > 0:
      frag_smi = fraginc2smi(incl, mol)
      f.n[idx].append([incl, frag_smi])
  h_atoms = [idx for idx, node in enumerate(f.n) if node[1] == 'H']
  for idx in sorted(h_atoms, reverse=True):
    f.rmnode(idx)
  if attr:
    for idx, node in enumerate(f.n):
      f.n[idx]=node_vec(node)
    for idx, edge in enumerate(f.e):
      f.e[idx]=bond_vec(edge)
    f.g=global_vec(Chem.MolToSmiles(mol))
    f_attr_dict=f.obj2dict()
  return f.obj2dict()

if __name__=="__main__":
  fn_ls=['inputs/gdb9_000039.xyz', 'inputs/gdb9_000658.xyz', 'inputs/gdb9_001216.xyz']
  gdb9_raw = [x.split(' ') for x in list(filter(None, open('inputs/GDB9_smi.txt', 'r').read().split('\n')))]
  gdb9_dict = dict()
  for x in gdb9_raw:
    gdb9_dict[x[0]]=x[1:]
  fn_ls=['inputs/gdb9_001216.xyz']
  #fn_ls=['inputs/gdb9_000039.xyz']
  #fn_ls=['inputs/gdb9_000658.xyz']
  for m in fn_ls:
    print('\n\n{}'.format(m))
    atoms, xyz_coordinates, mol = xyz2mol(m)
    f = mol2fraggraph(mol, m)
    fraginc_ls = cbh_n(f)
    for idx, incl in enumerate(fraginc_ls):
      if len(incl) > 0:
        frag_smi = fraginc2smi(incl, mol)
        f.n[idx].append([incl, frag_smi])
    h_atoms = [idx for idx, node in enumerate(f.n) if node[1] == 'H']
    for idx in sorted(h_atoms, reverse=True):
      f.rmnode(idx)
    pprint(f.obj2dict())
  print(f.n[0])
  print(f.e[0])
  print(node_vec(f.n[0]))
  print(bond_vec(f.e[0]))
  #sys.exit()
  #print(smi2mol(gdb9_dict['000039'][0]))
  '''
  f = mol2fraggraph(Chem.MolFromSmiles('c1ccccc1'),'c1ccccc1')
  pprint(f.obj2dict())
  '''


