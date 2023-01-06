import sys, os
from rdkit import Chem
from rdkit.Chem import Descriptors
import pprint
from mol2vec.mol2vec.features import mol2sentence, mol2alt_sentence, sentences2vec
from gensim.models import word2vec

def mol2atomicnum(mol):
  return [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()]

def mol2bondtypes(mol):
  return [float(bond.GetBondTypeAsDouble()) for bond in mol.GetBonds()]

def molwt(mol):
  return round(Chem.Descriptors.ExactMolWt(mol),8)

def num_atoms(mol):
  return mol.GetNumAtoms()

def num_heavy(mol):
  return len([x for x in mol2atomicnum(mol) if x != 1])

def num_rings(mol):
  return Chem.GetSSSR(mol)

def radElec(mol):
  return Chem.Descriptors.NumRadicalElectrons(mol)

def valElec(mol):
  return Chem.Descriptors.NumValenceElectrons(mol)

def enumerate_atoms(mol, atom_types=None):
  atoms = mol2atomicnum(Chem.AddHs(mol))
  if atom_types is None:
    atom_types = sorted(list(set(atoms)))
  else:
    atom_types = [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 31, 32, 33, 34, 35]
  return [atoms.count(x) for x in atom_types] 

def enumerate_bonds(mol, bond_types=None):
  bonds = mol2bondtypes(Chem.RemoveHs(mol))
  if bond_types is None:
    bond_types = sorted(list(set(bonds)))
  else:
    bond_types = [1.0, 1.5, 2.0, 3.0]
  return [bonds.count(x) for x in bond_types]

def dict2vec(d):
  vec=list()
  for k in sorted(d.keys()):
    val = d[k]
    if type(val) != list:
      val = [val]
    vec.extend(val)
  return vec

def smi2attr(smi):
  mol = Chem.MolFromSmiles(smi)
  mol = Chem.AddHs(mol)
  #print(pycbh.smi2mh(['propane', smi], 0))
  attr = dict()
  attr['atoms'] = enumerate_atoms(mol, atom_types='default')
  attr['bonds'] = enumerate_bonds(mol, bond_types='default')
  attr['molwt'] = molwt(mol)
  attr['num_atoms'] = num_atoms(mol)
  attr['num_heavy'] = num_heavy(mol)
  #attr['num_rings'] = num_rings(mol)
  #attr['radElec'] = radElec(mol)
  #attr['valElec'] = valElec(mol)
  return dict2vec(attr)

def smi2mol2vec(smi,model=None):
  
  if model is None:
    #try:
    if True:
      model = word2vec.Word2Vec.load('mol2vec/examples/models/model_300dim.pkl')
    #except:
    else:
      pass
  if model is not None:
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    vec= sentences2vec([mol2alt_sentence(mol,1)],model)[0].tolist()
    if type(vec) == list:
      if len(vec) == 300:
        return vec, model
  return [0.]*300, model

if __name__=='__main__':
  test_smi='[H][H]'
  attr = smi2attr(test_smi)
  vec, model = smi2mol2vec(test_smi)
  print(vec)
