import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader, Dataset
import torch_geometric.transforms as T
import copy
from math import sqrt

def load_model(model_path):
  saved = torch.load(model_path) #, map_location=torch.device("cuda"))
  model = saved['full_model']
  model.load_state_dict(saved["state_dict"])
  return model

def train(model, batches, num_iters=100, load_model=False, save_model=True, model_path='model_info.pth', lr=0.0001, test_batches=None):
  if load_model:
    saved = torch.load(model_path) #, map_location=torch.device("cuda"))
    model = saved['full_model']
    model.load_state_dict(saved["state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print('Loaded model and optimizer from {}'.format(model_path))
  else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


  model_best = model
  best_error = 1e10
  freq = 10
  for i in range(num_iters):
    running_loss = 0.
    count = 0
    for idx, data in enumerate(batches):
      optimizer.zero_grad()
      out, n_out = model(data)
      loss = F.mse_loss(out, data.y)
      loss.backward()
      running_loss += loss.detach() * out.size(dim=0)
      optimizer.step()
      count = count + out.size(dim=0)
    last_loss = running_loss / count
    if (i+1)%freq == 0 and test_batches is not None:
      y_pred_tr = predict(model, batches, 'Train ',return_mae=True)
      y_pred_te = predict(model, test_batches, 'Test - gdb9',return_mae=True)
      print('Loss ({:03d}) : {:.5f} | Train / Test : {:.5f} / {:.5f}'.format(i+1,last_loss,y_pred_tr,y_pred_te), flush=True)
    else:
      print('Loss ({:03d}) : {:.5f}'.format(i+1,last_loss), flush=True)
      
    if last_loss < best_error:
      model_best = copy.deepcopy(model)
      best_error = last_loss
      if save_model:
        torch.save({"state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "full_model": model}, model_path)

  return model_best

def mae(arr1, arr2):
  mean=0.0
  abso=0.0
  stdev=0.0
  diff = [xj-xi for xi, xj in zip(arr1,arr2)]
  for x in diff: 
    mean+=x
    abso+=abs(x)
  for x in diff: 
    stdev+=(x-mean/len(arr1))**2
  return round(mean/len(arr1),3), round(abso/len(arr1),3), round(1.96*sqrt(stdev/len(arr1)),3)

def predict(model, data, title, return_nodes=False, return_mae=False):
  if type(data) == list:
    data_ = DataLoader(data, batch_size=1, shuffle=False)
  else:
    data_ = data
  model.eval()
  pred_lst = [list(), list()]
  nodes_lst = list()
  for idx, g_i in enumerate(data_):
    with torch.no_grad():
      pred_i, pred_nodes = model(g_i)
      pred_nodes = pred_nodes.cpu().numpy().flatten()
      nodes_lst.extend(pred_nodes)
      pred_i = pred_i.cpu().numpy().flatten()
      targ_i = g_i.y.cpu().numpy().flatten()
      pred_lst[0].extend(pred_i)
      pred_lst[1].extend(targ_i)
  if return_mae:
    return mae(pred_lst[0],pred_lst[1])[1]
  print("  {} (N={}) : {}".format(title,len(pred_lst[0]),mae(pred_lst[0],pred_lst[1])))
  if return_nodes:
    pred_lst.append(nodes_lst)
  return pred_lst
