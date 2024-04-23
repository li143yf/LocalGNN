# LocalGNN

========

The code corresponds to article "Graph Partitioning for Lightweight GNNs: A Theoretical and Experimental Study", submitted to Neural Networks.
<p align="center">
  <img width="550" src="m1.jpg"> <img width="450" src="m2.jpg">
</p>

### Results

<p align="center">
  <img width="600" src="ruacc.jpg">  
</p>

<p align="center">
  <img width="300" src="clacc.jpg"> 
</p>


### Experimental environment 
All the experiments are conducted on a machine with NVIDIA GeForce RTX 3090GPU (24GB GPU memory). Intel Xeon Silver 4214R CPU(12 cores, 2.40GHz), and 256GB of RAM.



### Requirements
The codebase is implemented in Python 3.8.12. package versions used for development are just below.
```
torch              1.12.1
ogb                1.3.6
dgl                0.9.1
numpy              1.21.2
pandas             2.0.3
torchmetrics       1.0.3
```



### Datasets
<p align="justify">
 AmazonCoBuyComputer, CoauthorCS, CoauthorPhysics,  WikiCS, and Reddit dataset are from the dgl repository. Ogbn-arxiv is from https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv.</p>
 
### Graph Partitioning Algorithms
<p align="justify">
METIS, LDG, and FENNEL are provided by us in `util/li_Partitioner.py`. NE, DBH, and HDRF are from the original paper.</p>

### Explanation of the Document

util/Module.py includes GNN model.

util/li_Partitioner.py includes node partitioning algorithms to divide the local graph
   

### Examples
<p align="justify">
The following commands learn a neural network and score on the test set. </p>

```sh
$ python LocalGNN/localgnn_node.py
```



----------------------

