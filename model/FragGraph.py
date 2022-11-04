import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
import torch_geometric
from torch_geometric.nn import (
    Set2Set,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
    MetaLayer,
)
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter

class EdgeModel(torch.nn.Module):
    def __init__(self, dim, act):
        super(EdgeModel, self).__init__()
        self.act=act

        self.edge_mlp = torch.nn.ModuleList()

        self.edge_mlp.append(torch.nn.Linear(dim * 2, 128))
        self.edge_mlp.append(torch.nn.Linear(128,128))
        self.edge_mlp.append(torch.nn.Linear(128,64))
        self.edge_mlp.append(torch.nn.Linear(64,dim))

    def forward(self, src, dest, edges, globals, batch):
        comb = torch.cat([src, edges], dim=1)
        for i in range(0, len(self.edge_mlp)):
            if i == 0:
                out = self.edge_mlp[i](comb)
                out = getattr(F, self.act)(out)
            else:
                out = self.edge_mlp[i](out)
                out = getattr(F, self.act)(out)
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, dim, act):
        super(NodeModel, self).__init__()
        self.act=act

        self.node_mlp = torch.nn.ModuleList()

        self.node_mlp.append(torch.nn.Linear(dim * 2, 128))
        self.node_mlp.append(torch.nn.Linear(128,128))
        self.node_mlp.append(torch.nn.Linear(128,64))
        self.node_mlp.append(torch.nn.Linear(64,dim))

    def forward(self, nodes, edge_index, edges, globals, batch):
        v_e = scatter_mean(edges, edge_index[0, :], dim=0)
        comb = torch.cat([nodes, v_e], dim=1)
        for i in range(0, len(self.node_mlp)):
            if i == 0:
                out = self.node_mlp[i](comb)
                out = getattr(F, self.act)(out)
            else:
                out = self.node_mlp[i](out)
                out = getattr(F, self.act)(out)
        return out

class FragGraph(torch.nn.Module):
    def __init__(self, data, dim1=128, dim2=128, dim3=64, deg=None, node_model='mean'):
       super(FragGraph, self).__init__()
        self.pool_reduce="mean"
        self.act = 'relu'

        recurrent_passes = 3
        output_dim = 1

        self.num_features = {'edges': data.edges.size(dim=-1), 'nodes': data.nodes.size(dim=-1), 'globals': data.globals.size(dim=-1)}
        self.latent_size  = {'edges': 256, 'nodes': 256, 'globals':1}
        print('self.num_features',self.num_features)
        print('self.latent_size',self.latent_size)

        self.encode_edges = torch.nn.ModuleList()
        self.encode_nodes = torch.nn.ModuleList()
        self.encode_globals = torch.nn.ModuleList()

        self.encode_edges.append(Sequential(
                                    Linear(self.num_features['edges'], dim1), ReLU(),
                                    Linear(dim1,dim2), ReLU(),
                                    Linear(dim2,dim3), ReLU(),
                                    Linear(dim3, self.latent_size['edges']),
                                 ))
        self.encode_nodes.append(Sequential(
                                    Linear(self.num_features['nodes'], dim1), ReLU(),
                                    Linear(dim1,dim2), ReLU(),
                                    Linear(dim2,dim3), ReLU(),
                                    Linear(dim3, self.latent_size['nodes']),
                                 ))
        self.encode_globals.append(Sequential(Linear(self.num_features['globals'], self.latent_size['globals'])))

        ##Set up GNN layers
        self.conv_list = torch.nn.ModuleList()
        for i in range(recurrent_passes):
            self.conv_list.append(
                MetaLayer(
                    EdgeModel(self.latent_size['edges'], self.act),
                    NodeModel(self.latent_size['nodes'], self.act),
                )
            )


        self.decode_nodes = torch.nn.ModuleList()
        self.decode_nodes.append(Sequential(
                                    Linear(self.latent_size['nodes'], dim1), ReLU(),
                                    Linear(dim1,dim2), ReLU(),
                                    Linear(dim2,dim3), ReLU(),
                                    Linear(dim3, 1)
                                 ))
        ##Pre-GNN dense layers
        e_out = self.encode_edges[0](data.edges)
        n_out = self.encode_nodes[0](data.nodes)
        g_out = self.encode_globals[0](data.globals)

        ##GNN layers
        for i in range(0, len(self.conv_list)):
            n_temp, e_temp, g_temp = self.conv_list[i](
                n_out, data.edge_index, e_out, g_out, data.batch
                )
            n_out = torch.add(n_out, n_temp)
            e_out = torch.add(e_out, e_temp)

        n_out = self.decode_nodes[0](n_out)

        out = torch_geometric.nn.global_add_pool(n_out, data.batch)


        return out, n_out
