import sys, os
from utils import *

def prepare_inputs():
  xyz_ls = ['inputs/gdb9_000039.xyz', 'inputs/gdb9_000658.xyz', 'inputs/gdb9_001216.xyz']
  smi_ls = [x.split(' ') for x in list(filter(None, open('inputs/GDB9_smi.txt', 'r').read().split('\n')))]
  return xyz_ls, smi_ls

if __name__=="__main__":

  xyz_ls, smi_ls = prepare_inputs()
    
  for m in xyz_ls:
    print(m)
    f = xyz2fraggraph(m,attr=False) #True)
    
  for m in smi_ls:
    print(m)
    f = smi2fraggraph(m[1],attr=True)

  print(f)


