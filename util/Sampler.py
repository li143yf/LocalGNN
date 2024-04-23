from dgl import DGLError, set_node_lazy_features, set_edge_lazy_features
from dgl.dataloading import Sampler
import os
import pickle
import dgl
import numpy as np
import torch

class Universal_Sampler(Sampler):
    def __init__(self, g, k, cache_path, partition_file_path=None, prefetch_ndata=None,
                 prefetch_edata=None, output_device=None):      
        super().__init__()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.partition_node_ids, self.partition_offset = pickle.load(f)
            except (EOFError, TypeError, ValueError):
                raise DGLError(
                    f'The contents in the cache file {cache_path} is invalid. '
                    f'Please remove the cache file {cache_path} or specify another path.')      
        else:
            partition_offset = []
            partition_node_ids = []
            partition_offset.append(0)
            node_partition = {}
    
            with open(partition_file_path, 'r') as f:
                i = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    classes = int(line)
                    if classes not in node_partition:
                        node_partition[classes] = []
                    node_partition[classes].append(i)
                    i += 1        
            for i in range(k):
                partition_node_ids += node_partition[i]
                partition_offset.append(partition_offset[-1] + len(partition_node_ids))
            partition_offset = torch.tensor(partition_offset)
            partition_node_ids = torch.tensor(partition_node_ids)          
            with open(cache_path, 'wb') as f:
                pickle.dump((partition_node_ids, partition_offset), f)
            self.partition_offset = partition_offset
            self.partition_node_ids = partition_node_ids

        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []
        self.output_device = output_device


class Node_partition_sampler(Universal_Sampler):
    def __init__(self, g, k, cache_path, partition_file_path=None, prefetch_ndata=None,
                 prefetch_edata=None, output_device=None):
        super(Node_partition_sampler, self).__init__(g, k, cache_path, partition_file_path,
                                                     prefetch_ndata, prefetch_edata, output_device)

    def sample(self, g, partition_ids):
        node_ids = torch.cat([
            self.partition_node_ids[self.partition_offset[i]:self.partition_offset[i + 1]]
            for i in partition_ids], 0)
        sg = g.subgraph(node_ids, relabel_nodes=True, output_device=self.output_device)
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return sg


def load_block_from_file(filename):
    with open(filename, 'rb') as file:
        block = pickle.load(file)
    return block

class hrdf_Sampler(Sampler):
    def __init__(self, g, k, cache_path, partition_file_path=None, prefetch_ndata=None,
                 prefetch_edata=None, output_device=None):
        super().__init__()
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.blocks = pickle.load(f) 
            except (EOFError, TypeError, ValueError):
                raise DGLError(
                    f'The contents in the cache file {cache_path} is invalid. '
                    f'Please remove the cache file {cache_path} or specify another path.')
        else:
             pass
        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []
        self.output_device = output_device


class Edge_partition_sampler(hrdf_Sampler):
    def __init__(self, g, k, cache_path, partition_file_path=None, prefetch_ndata=None,
                 prefetch_edata=None, output_device=None):
        super(Edge_partition_sampler, self).__init__(g, k, cache_path, partition_file_path,
                                                     prefetch_ndata, prefetch_edata, output_device)
        
    def sample(self, g, partition_ids): 
        merged_set = set()
        for i in partition_ids:
             if i <= len(self.blocks) and self.blocks[i] is not None:  
                 merged_set = merged_set.union(self.blocks[i])
             else:
                 raise ValueError("over-long-blocks,kong-block.") 
        src = []
        dst = []
        for my_tuple in merged_set:
            v1, v2 = my_tuple            
            src.append(int(v1))
            dst.append(int(v2))
        x, y = src,dst
        if len(x) != len(y):
            raise ValueError("The length of src and dst should be the same.")
        edge_id_list = []
        for i in range(len(x)):
            edge_id = g.edge_ids(int(x[i]), int(y[i]))
            edge_id_list.append(edge_id)
        sg = dgl.edge_subgraph(g, edge_id_list,relabel_nodes=True, output_device=self.output_device)
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return sg


