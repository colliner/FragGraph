import sys
import os
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Load and process graph data.')
    parser.add_argument('--data_path',
                        type=str,
                        default='data/gdb9_fragment_embedded_graphs',
                        help='Path to the data directory.')
    return parser.parse_args()


def load_data(files):
    """
    Load graph data from files.
    
    Args:
        files (list): List of file paths.
    
    Returns:
        list: List of loaded graphs.
    """
    graphs = list()
    np_load_old = np.load

    # Override np.load to allow loading pickled data
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    for i, fn in enumerate(files):
        print('Loading file {:02d} / {:02d} : {}'.format(i + 1, len(files), fn))

        # Load graph data from file and extend the graphs list
        graphs.extend(np.load(fn).tolist())

    # Restore the original np.load function
    np.load = np_load_old
    return graphs


def load_pop(pop_file):
    """
    Load population data from file.
    
    Args:
        pop_file (str): Path to the population file.
    
    Returns:
        dict: Dictionary of population data.
    """
    np_load_old = np.load

    # Override np.load to allow loading pickled data
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

    # Load population data from file
    pop_dict = np.load(pop_file).tolist()[0]

    # Restore the original np.load function
    np.load = np_load_old
    print('Loaded {} pops from {}'.format(len(pop_dict.keys()), pop_file))
    return pop_dict


def load_split_fn(fns):
    """
    Load split data from files.
    
    Args:
        fns (list or str): List of file paths or a single file path.
    
    Returns:
        dict: Dictionary of split data.
    """
    if type(fns) != list:
        # Convert single file path to a list
        fns = [fns]
    splits = []
    for fn in fns:
        try:
            # Load split data from file
            splits.extend([
                x.split(" ")
                for x in list(filter(None,
                                     open(fn, "r").read().split("\n")))
            ])
        except:
            print('could not find', fn)
            continue
    split_dict = dict()
    for x in splits:
        # Create a dictionary of split data
        split_dict[x[0].split('/')[-1].split('.')[0]] = x[1]
    return split_dict


def graph2torch(graph, y=None, fc=True, pop_nodes=None):
    """
    Convert graph to PyTorch Data object.
    
    Args:
        graph (dict): Graph data dictionary.
        y (float, optional): Target value. Defaults to None.
        fc (bool, optional): Flag for fully connected graph. Defaults to True.
        pop_nodes (list, optional): Population nodes. Defaults to None.
    
    Returns:
        torch_geometric.data.Data: PyTorch Data object.
    """

    # Convert nodes to tensor
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
    if len(graph['edges']) > 0:  # mol has bonds
        # Convert edges to tensor
        edges = torch.tensor(np.asarray(graph['edges']), dtype=torch.float32)

        # Convert edge indices to tensor
        edge_index = torch.tensor(np.asarray(
            [graph['senders'], graph['receivers']]),
                                  dtype=torch.long)
    else:
        return None

    # Convert globals to tensor
    g = torch.tensor(np.asarray([graph['globals']]), dtype=torch.float32)

    # Convert target value to tensor
    y = torch.tensor(np.array([[y]]), dtype=torch.float32)
    atom_types = list()
    for node in graph['nodes']:
        atom_types.append(node[:6])

    # Convert atom types to tensor
    atom_types = torch.tensor(np.asarray(atom_types), dtype=torch.float32)
    if pop_nodes is not None:
        # Convert population nodes to tensor
        pop_nodes_ = torch.tensor(np.asarray(pop_nodes), dtype=torch.float32)

        # Create PyTorch Data object with population nodes
        data = Data(nodes=nodes,
                    edge_index=edge_index,
                    edges=edges,
                    globals=g,
                    y=y,
                    pop_nodes=pop_nodes_,
                    atom_types=atom_types)
    else:
        # Create PyTorch Data object without population nodes
        data = Data(nodes=nodes,
                    edge_index=edge_index,
                    edges=edges,
                    globals=g,
                    y=y,
                    atom_types=atom_types)
    return data


def load_torch_from_dicts(
        files,
        y_dict,
        splits=None,
        fc=True,
        override_IP=False,
        pop=None,
        override_global=False,
        global_fn='data/esp_wb97xd_large_no_collapse_w_frag_vec.txt',
        simple_graphs=False,
        use_zero_pop=False):
    """
    Load PyTorch Data objects from dictionaries.
    
    Args:
        files (list or str): List of file paths or a single file path.
        y_dict (dict): Dictionary of target values.
        splits (dict, optional): Dictionary of split data. Defaults to None.
        fc (bool, optional): Flag for fully connected graph. Defaults to True.
        override_IP (bool, optional): Flag for overriding IP. Defaults to False.
        pop (dict, optional): Dictionary of population data. Defaults to None.
        override_global (bool, optional): Flag for overriding global values. Defaults to False.
        global_fn (str, optional): Path to the global file. Defaults to 'data/esp_wb97xd_large_no_collapse_w_frag_vec.txt'.
        simple_graphs (bool, optional): Flag for simple graphs. Defaults to False.
        use_zero_pop (bool, optional): Flag for using zero population. Defaults to False.
    
    Returns:
        tuple: Tuple containing training and testing graph lists and label lists.
    """
    if type(files) != list:
        # Convert single file path to a list
        files = [files]
    if type(splits) != dict:
        # Initialize empty splits dictionary if not provided
        splits = {}

    # Load graph data from files
    graphs = load_data(files)

    # Initialize lists for training and testing graphs
    graphs_lst_tr, graphs_lst_te = list(), list()
    pop_err = True

    # Initialize lists for training and testing labels
    labels_lst_tr, labels_lst_te = list(), list()
    global_d = dict()
    if override_global:
        # Load global data from file
        inlines = [
            x.split(' ')
            for x in list(filter(None,
                                 open(global_fn, 'r').read().split('\n')))
        ]
        for x in inlines:
            # Create a dictionary of global data
            global_d[x[0]] = x[1:]
    for g in graphs:
        # Extract label from graph data
        label = g['label'].split('/')[-1].split('.')[0]
        if pop is not None:
            sys.exit()
            label_ = label.replace('_neu', '').replace('_cat', '')
            if label_ in pop:
                pop_nodes = list()
                pop_err = False
                pop_lst = pop[label_]
                node_idx = 0
                for pop_idx, pop_d in enumerate(pop_lst):
                    if pop_d[0] == 1:
                        continue
                    else:
                        if g['nodes'][node_idx][[6, 7, 8, 9, 16,
                                                 17].index(pop_d[0])] == 1.0:
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
            # Override global values with data from the global dictionary
            g['globals'] = [float(x_) for x_ in global_d[label]]

        if override_IP:
            if 'qm7b' in label:
                # Override IP for qm7b labels
                label = label.replace('qm7b_', 'qm7b_IP_')
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
                # Convert graph to PyTorch Data object with population nodes
                torch_data = graph2torch(g,
                                         y_dict[label],
                                         fc=fc,
                                         pop_nodes=pop_nodes)
            else:
                # Convert graph to PyTorch Data object without population nodes
                torch_data = graph2torch(g, y_dict[label], fc=fc)
            if torch_data is not None:
                if label not in splits:
                    new_label_ = label.replace('_cat', '').replace('_neu', '')
                    if new_label_ in splits:
                        label = new_label_
                if label in splits:
                    if splits[label] == 'True':
                        # Add graph to testing list
                        graphs_lst_te.append(torch_data)

                        # Add label to testing list
                        labels_lst_te.append(label)
                        continue

                # Add graph to training list
                graphs_lst_tr.append(torch_data)

                # Add label to training list
                labels_lst_tr.append(label)
        else:
            print('ERROR: {} not in y_dict'.format(label))

    # Return training and testing graph and label lists
    return graphs_lst_tr, graphs_lst_te, labels_lst_tr, labels_lst_te


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Get the path to the data directory from command-line arguments
    path_to_gdb9_attr = args.data_path

    # Construct file paths
    files = [
        f'{path_to_gdb9_attr}/gdb9_skeleton_graphs_nocg_0a_attr300.npy',
        f'{path_to_gdb9_attr}/gdb9_skeleton_graphs_nocg_0b_attr300.npy'
    ]

    # Load graph data from files
    graphs = load_data(files)

    # Convert graphs to PyTorch Data objects
    torchs = [graph2torch(graph, 1.0) for graph in graphs]
    print(len(graphs), len(torchs))
    for g, t_g in zip(graphs, torchs):
        if t_g is not None:
            print(g)
            for k, v in g.items():
                print(k, np.asarray(v).shape)
            print(t_g)
            break
