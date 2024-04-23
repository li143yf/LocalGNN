import random
from abc import abstractmethod
import torch
import math
import pickle
import numpy as np
import dgl
import dgl.backend as F
import time

class Partitioner:
    def __init__(self, graph, k):
        self.graph = graph
        self.k = k
        self.partition_node_ids = None
        self.partition_offset = None
        self.blocks = None
        pass

    @abstractmethod
    def divide(self):

        pass

    def save(self, path):
        if self.partition_node_ids is None or self.partition_offset is None:
            self.divide()

        print("save to", path)
        with open(path, 'wb') as f:
            pickle.dump((self.partition_node_ids, self.partition_offset), f)
        pass

    def save_hdrf(self, path):
        if self.blocks is None:
            self.divide()
        print("save to", path)
        with open(path, 'wb') as f:
            pickle.dump((self.blocks), f)
        pass



class LDG(Partitioner):
    def __init__(self, graph, k, beta=1.1):
        super().__init__(graph, k)
        assert beta >= 1
        self.c = math.ceil(beta * graph.num_nodes() / k)

    def divide(self):
        t0 = time.time()
        nodes = {}
        edge = self.graph.edges()
        e1 = edge[0].numpy().tolist()
        e2 = edge[1].numpy().tolist()
        for i in range(self.graph.num_edges()):
            u, v = e1[i], e2[i]
            if u != v:  
                if u not in nodes:
                    nodes[u] = set()
                if v not in nodes:
                    nodes[v] = set()
                nodes[u].add(v)
                nodes[v].add(u)
        block = [set() for _ in range(self.k)]
        block_score = [0 for _ in range(self.k)]
        random.seed(123)
        nodes_list = list(range(self.graph.num_nodes()))
        random.shuffle(nodes_list)
        a=0
        for node in nodes_list:
            if node in nodes:
                a+=1
                for i in range(self.k):
                    w = 1 - len(block[i]) / self.c
                    block_score[i] = w * (len(nodes[node] & block[i]) + 1)
                block[block_score.index(max(block_score))].add(node)
                pass

        num_nodes_list = []
        num_edges_list = []
        for subset in block:
            int_elements = [int(x) for x in subset]
            node_ids = torch.tensor(int_elements)
            minisg = self.graph.subgraph(node_ids, relabel_nodes=True)
            num_nodes = minisg.num_nodes()
            num_edges = minisg.num_edges()
            num_nodes_list.append(num_nodes)
            num_edges_list.append(num_edges)

        sum_of_nodes = sum(num_nodes_list)
        average_of_nodes = sum(num_nodes_list) / len(num_nodes_list)
        sum_of_edges = sum(num_edges_list)
        average_of_edges = sum(num_edges_list) / len(num_edges_list)

        sum_of_edges_all=self.graph.num_edges()
        cut=1-sum_of_edges/sum_of_edges_all
        partition_node_ids = torch.cat(tuple(map(lambda x: torch.tensor(list(x)), block)), dim=0)
        partition_offset = [0]
        for i in range(len(block)):
            partition_offset.append(partition_offset[-1] + len(block[i]))
        partition_offset = torch.tensor(partition_offset)
        t1 = time.time()
        huafentime=t1-t0
        self.partition_node_ids = partition_node_ids
        self.partition_offset = partition_offset
        return partition_node_ids, partition_offset




class Fennel(Partitioner):
    def __init__(self, graph, k, nu=1.1, gamma=1.5):
        super().__init__(graph, k)
        self.gamma = gamma
        self.load_limit = nu * graph.num_nodes() / k
        self.alpha = math.sqrt(k) * graph.num_edges() / math.pow(graph.num_nodes(), 1.5)

    def divide(self):
        t0 = time.time()
        nodes = {}
        edge = self.graph.edges()
        e1 = edge[0].numpy().tolist()
        e2 = edge[1].numpy().tolist()
        for i in range(self.graph.num_edges()):
            u, v = e1[i], e2[i]
            if u != v:  
                if u not in nodes:
                    nodes[u] = set()
                if v not in nodes:
                    nodes[v] = set()
                nodes[u].add(v)
                nodes[v].add(u)
        block = [set() for _ in range(self.k)]
        block_score = [0 for _ in range(self.k)]
        random.seed(123)
        nodes_list = list(range(self.graph.num_nodes()))
        random.shuffle(nodes_list)
        a=0
        for node in nodes_list:
            if node in nodes:
                a+=1
                for i in range(self.k):
                    if len(block[i]) < self.load_limit:
                        block_score[i] = len(nodes[node] & block[i]) - self.alpha * self.gamma * math.pow(len(block[i]), self.gamma - 1)
                    else:                             
                        block_score[i] = -999999999999

                block[block_score.index(max(block_score))].add(node)
                pass   
        num_nodes_list = []
        num_edges_list = []
        for subset in block:
            int_elements = [int(x) for x in subset]
            node_ids = torch.tensor(int_elements)
            minisg =self.graph.subgraph(node_ids, relabel_nodes=True)
            num_nodes = minisg.num_nodes()
            num_edges = minisg.num_edges()
            num_nodes_list.append(num_nodes)
            num_edges_list.append(num_edges)
        sum_of_nodes = sum(num_nodes_list)
        average_of_nodes = sum(num_nodes_list) / len(num_nodes_list)
        sum_of_edges = sum(num_edges_list)
        average_of_edges = sum(num_edges_list) / len(num_edges_list)
        sum_of_edges_all=self.graph.num_edges()
        cut=1-sum_of_edges/sum_of_edges_all
        partition_node_ids = torch.cat(tuple(map(lambda x: torch.tensor(list(x)), block)), dim=0)
        partition_offset = [0]

        block_nodenum=[]
        for i in range(len(block)):
            x=len(block[i])
            block_nodenum.append(x)

        for i in range(len(block)):
            partition_offset.append(partition_offset[-1] + len(block[i]))
        partition_offset = torch.tensor(partition_offset)
        t1 = time.time()
        huafentime=t1-t0
        self.partition_node_ids = partition_node_ids
        self.partition_offset = partition_offset
        return partition_node_ids, partition_offset


class Metis(Partitioner):
    def __init__(self, graph, k):
        super().__init__(graph, k)
        self.g = graph
        self.k = k

    def divide(self):
        t0 = time.time()  
        partition_ids = dgl.metis_partition_assignment(
            self.g, self.k, mode="k-way")
        partition_ids = F.asnumpy(partition_ids)
        partition_node_ids = np.argsort(partition_ids)
        partition_size = F.zerocopy_from_numpy(np.bincount(partition_ids, minlength=self.k))
        partition_offset = F.zerocopy_from_numpy(np.insert(np.cumsum(partition_size), 0, 0))
        partition_node_ids = F.zerocopy_from_numpy(partition_node_ids)
        self.partition_node_ids = partition_node_ids
        self.partition_offset = partition_offset
        t1 = time.time()
        huafentime = t1 - t0
        num_nodes_list = []
        num_edges_list = []
        for i in range(0,self.k):
            node_ids = self.partition_node_ids[self.partition_offset[i]:self.partition_offset[i + 1]]
            node_ids = torch.tensor(node_ids)
            minsg = self.g.subgraph(node_ids, relabel_nodes=True)  
            num_nodes = minsg.num_nodes()
            num_edges = minsg.num_edges()
            num_nodes_list.append(num_nodes) 
            num_edges_list.append(num_edges)
        sum_of_nodes = sum(num_nodes_list)
        average_of_nodes = sum(num_nodes_list) / len(num_nodes_list)
        sum_of_edges = sum(num_edges_list)
        average_of_edges = sum(num_edges_list) / len(num_edges_list)
        sum_of_edges_all = self.g.num_edges()
        cut = 1 - sum_of_edges / sum_of_edges_all  
        return partition_node_ids, partition_offset
        pass


