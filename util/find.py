import pickle
import numpy as np
import torch
import random
import dgl

def f_co_el(a, b):

    a_np = a.numpy()
    b_np = b.numpy()

    common_elements = np.intersect1d(a_np, b_np)
    positions_a = []
    positions_b = []
    for element in common_elements:
        pos_a = np.where(a_np == element)[0][0]
        pos_b = np.where(b_np == element)[0][0]

        positions_a.append(pos_a)
        positions_b.append(pos_b)
    return positions_a, positions_b


def f_t_co_el(a, b):
    a_np = np.array(a)
    b_np = b.numpy()
    common_elements = np.intersect1d(a_np, b_np)

    positions_a = []
    train_common_element=[]
    for element in common_elements:
        pos_a = np.where(a_np == element)[0][0]
        train_common_element.append(element)

        positions_a.append(pos_a)

    return train_common_element, positions_a


def m_sub (g,list_block_id,cache_path) :
    merged_set = set()
    with open(cache_path, 'rb') as f:
        blocks = pickle.load(f)
    for i in list_block_id:
        merged_set = merged_set.union(blocks[i])
    src = []
    dst = []
    for my_tuple in merged_set:
        v1, v2 = my_tuple
        src.append(int(v1))
        dst.append(int(v2))
    x, y = src, dst
    if len(x) != len(y):
        raise ValueError("The length of src and dst should be the same.")

    edge_id_list = []

    for i in range(len(x)):
        edge_id = g.edge_ids(int(x[i]), int(y[i]))
        edge_id_list.append(edge_id)

    sg = dgl.edge_subgraph(g, edge_id_list, relabel_nodes=True)
    sg = sg.to('cuda:0')
    return sg

def cosine(x1, x2,temperature,eps=1e-15):

    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    dot_product1 = torch.sum(x1 * x2, dim=1)
    dot_product2 = torch.sum(w1 * w2, dim=1)
    cosine_similarity = dot_product1 / (dot_product2.clamp(min=eps) * temperature)
    cosine_similarity = torch.exp(cosine_similarity)
    return torch.sum(cosine_similarity)
