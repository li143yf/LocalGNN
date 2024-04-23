import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import util.Sampler
import util.Module
import util.find
import csv
import random
import time
import math
import pickle
from ogb.nodeproppred import DglNodePropPredDataset


methods = ["DBH", "HDRF", "NE"]  
method = methods[1]  
num_partitions = 100
num_epochs = 100
batch_size = 1
num_hidden = 128
lr = 0.001
weight_decay =  5e-4
dropout = 0.5



aveLOSS = []
aveTrain_acc = []
Val_acc = []
Test_acc = []
node_num = []
edge_num = []


dataset = dgl.data.AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root="./data/arxiv"))
graph = dataset[0]
num_edges = graph.num_edges()
num_nodes = graph.num_nodes()


cl_tensor = np.zeros((num_epochs, num_partitions), dtype=int)
for i in range(num_epochs):
    row = np.arange(num_partitions)
    np.random.shuffle(row)
    cl_tensor[i] = row
cl_tensor = torch.from_numpy(cl_tensor)
model = util.Module.SAGE(graph.ndata["feat"].shape[1], num_hidden, dataset.num_classes).cuda()
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
eval_dataloader = dgl.dataloading.DataLoader(
    graph,
    torch.arange(num_partitions).to("cuda"),
    util.Sampler.Edge_partition_sampler(
        graph,
        num_partitions,
        cache_path=f'./data/arxhdrf',   
        prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],

    ),
    device="cuda",
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    use_uva=True,
)
durations = []
be_cl=3
for e in range(num_epochs):
    if e < be_cl:       
        cl_line = e 
        cl_dataloader_tensor = cl_tensor[cl_line]
        dataloader = dgl.dataloading.DataLoader(
            graph,
            cl_dataloader_tensor.to("cuda"),
            util.Sampler.Edge_partition_sampler(
                graph,
                num_partitions,
                cache_path=f'./data/arxhdrf',   
                prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
            ),
            device="cuda",
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            use_uva=True,
        )
        t0 = time.time()
        model.train()
        i = 0
        Loss = []
        Acc = []
        for it, sg in enumerate(dataloader):
            i += 1
            x = sg.ndata["feat"]
            y = sg.ndata["label"]
            m = sg.ndata["train_mask"].bool()
            y_hat, b = model(sg, x)
            loss = F.cross_entropy(y_hat[m], y[m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            Loss.append(loss.item())

            acc = MF.accuracy(y_hat[m], y[m], task="multiclass",
                              num_classes=dataset.num_classes)
            Acc.append(acc.item())
            
        aveLOSS.append(f"{sum(Loss) / i:.8f}")
        aveTrain_acc.append(f"{sum(Acc) / i:.8f}")
        tt = time.time()
        durations.append(tt - t0)


        model.eval()
        with torch.no_grad():     
            val_preds, test_preds = [], []
            val_labels, test_labels = [], []
            for it, sg in enumerate(eval_dataloader): 
                x = sg.ndata["feat"] 
                y = sg.ndata["label"] 
                m_val = sg.ndata["val_mask"].bool()
                m_test = sg.ndata["test_mask"].bool()
                y_hat = model.inference(sg, x)
                val_preds.append(y_hat[m_val])
                val_labels.append(y[m_val])
                test_preds.append(y_hat[m_test])
                test_labels.append(y[m_test])
            val_preds = torch.cat(val_preds, 0)  
            val_labels = torch.cat(val_labels, 0)
            test_preds = torch.cat(test_preds, 0)
            test_labels = torch.cat(test_labels, 0)

            val_acc = MF.accuracy(val_preds, val_labels, task="multiclass", num_classes=dataset.num_classes)
            test_acc = MF.accuracy(test_preds, test_labels, task="multiclass", num_classes=dataset.num_classes)

            print(f"Val Acc {val_acc.item():.4f} | Test Acc {test_acc.item():.4f}")
            Val_acc.append(val_acc.item())
            Test_acc.append(test_acc.item())
    else:
        cl_line = e  
        cl_dataloader_tensor = cl_tensor[cl_line]
        cl_dataloader = dgl.dataloading.DataLoader(
            graph,
            cl_dataloader_tensor.to("cuda"),
            util.Sampler.Edge_partition_sampler(
                graph,
                num_partitions,
                cache_path=f'./data/arxhdrf',    
                prefetch_ndata=["feat", "label", "train_mask", "val_mask", "test_mask"],
            ),
            device="cuda",
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            use_uva=True,
        )
        model.train()
        positive_sg_y_hat = []
        original_id_list = []
        y_label = []
        Loss = []
        Acc = []
        t0 = time.time()
      
        for it, sg in enumerate(cl_dataloader): 
            if it == 0:
                f_e = cl_dataloader_tensor[0]
            elif it == 1: 
                s_e = cl_dataloader_tensor[1]
            elif it == 2: 
                t_e = cl_dataloader_tensor[2]
                cache_path = f'./data/arxhdrf'
                sg = util.find.m_sub(graph,[f_e,s_e,t_e],cache_path)
                x = sg.ndata["feat"]
                y1 = sg.ndata["label"]
                m = sg.ndata["train_mask"].bool()
                true_indices1 = torch.nonzero(m, as_tuple=True)[0]   
                y_hat_1, b1 = model(sg, x)
                loss = F.cross_entropy(y_hat_1[m], y1[m])
                opt.zero_grad()
                loss.backward()
                opt.step()
                Loss.append(loss.item())
                acc = MF.accuracy(y_hat_1[m], y1[m], task="multiclass",
                                  num_classes=dataset.num_classes)
                Acc.append(acc.item())

                y1_label = y1.tolist()  
                y1_label_1di = {}
                for i, num in enumerate(y1_label):
                    if num not in y1_label_1di:
                        y1_label_1di[num] = []
                    y1_label_1di[num].append(i)  
                original_id_1 = sg.ndata['_ID']              
                positive_sg_b1 = b1.tolist()  
                positive_sg_y_hat_1 = y_hat_1.tolist()  
            else:
                x = sg.ndata["feat"]
                yx = sg.ndata["label"]
                m = sg.ndata["train_mask"].bool()
                true_indicesx = torch.nonzero(m, as_tuple=True)[0]               
                y_hat_x, bx = model(sg, x)
                original_id_x = sg.ndata['_ID']
                original_id_x = original_id_x.cpu()
                y_label_x = yx.tolist()
                positive_sg_bx = bx.tolist()  
                positive_sg_y_hat_x = y_hat_x.tolist()   
                allcl_po = []
                allcl_po_x = []
                allcl_ne = []
                allcl_ne_x = []
                allcl_po1 = []
                allcl_po_x1 = []
                allcl_ne1 = []
                allcl_ne_x1 = []

                yx_label_2 = yx.tolist()
                yx_label_2di = {}
                for i, num in enumerate(yx_label_2): 
                    if num not in yx_label_2di:
                        yx_label_2di[num] = []
                    yx_label_2di[num].append(i)
                batch_cl_loss_1 = 0  
                original_id_1 =original_id_1.cpu()
                index_a, index_b = util.find.f_co_el(original_id_1, original_id_x)
                true_indices1 = true_indices1.cpu()
                list_a,list_b= util.find.f_t_co_el(index_a,true_indices1) 
                if len(list_a) == 0:   
                    comm = 0
                else:
                    ture_list_b= [index_b[i] for i in list_b]
                    num_listab = 0
                    
                    for i in list_a :
                         row_1_1 = positive_sg_b1[i]   
                         row_1_x = positive_sg_bx[ture_list_b[num_listab]]
                         row_1 = positive_sg_y_hat_1[i]  
                         row_x = positive_sg_y_hat_x[ture_list_b[num_listab]]
                         if row_1.index(max(row_1)) == int(y1[i]) or row_x.index(max(row_x)) == int(yx[ture_list_b[num_listab]]):  
                             allcl_po.append(row_1)
                             allcl_po_x.append(row_x)
                             allcl_po1.append(row_1_1)
                             allcl_po_x1.append(row_1_x)
                         else:
                             allcl_ne.append(row_1)
                             allcl_ne_x.append(row_x)
                             allcl_ne1.append(row_1_1)
                             allcl_ne_x1.append(row_1_x)
                         num_listab +=1                   
                    true_indicesx = true_indicesx.cpu()
                    true_indicesx_list= true_indicesx.tolist()
                    random_x_id = random.sample(true_indicesx_list, len(allcl_po))
                    for x_id in random_x_id:
                        po_label = y_label_x[x_id]
                        ne_index = random.choice([value for key, value in y1_label_1di.items() if key != "po_label"])
                        sg1_id = random.choice(ne_index)
                        row_1 = positive_sg_y_hat_1[sg1_id]
                        row_x = positive_sg_y_hat_x[x_id]
                        row_1_1 = positive_sg_b1[sg1_id]
                        row_1_x = positive_sg_bx[x_id]
                        allcl_ne.append(row_1)
                        allcl_ne_x.append(row_x)
                        allcl_ne1.append(row_1_1)
                        allcl_ne_x1.append(row_1_x)
                temperature = 0.2
                cos_po = util.find.cosine(torch.tensor(allcl_po), torch.tensor(allcl_po_x),temperature)
                if len(allcl_ne) == 0 :
                      cos_ne = 10 
                else:
                      cos_ne = util.find.cosine(torch.tensor(allcl_ne), torch.tensor(allcl_ne_x),temperature)
                allcl_po1 = torch.tensor(allcl_po1)
                allcl_po1 = F.normalize(allcl_po1, dim=1)
                allcl_po_x1 = torch.tensor(allcl_po_x1)
                allcl_po_x1 = F.normalize(allcl_po_x1, dim=1)
                allcl_ne1 = torch.tensor(allcl_ne1)
                allcl_ne1 = F.normalize(allcl_ne1, dim=1)
                allcl_ne_x1 = torch.tensor(allcl_ne_x1)
                allcl_ne_x1 = F.normalize(allcl_ne_x1, dim=1)
                cos_po1 = util.find.cosine(allcl_po1, allcl_po_x1,temperature)
                if len(allcl_ne) == 0:  
                    cos_ne1 = 10  
                else:
                    cos_ne1 = util.find.cosine(allcl_ne1, allcl_ne_x1,temperature)
                
                batch_cl_loss = -torch.log( cos_po1/ (cos_po1 + cos_ne1))
                alpha = 1
                loss = F.cross_entropy(y_hat_x[m], yx[m]).to("cuda") + alpha * batch_cl_loss.to("cuda")
                opt.zero_grad()
                loss.backward()
                opt.step()
                Loss.append(loss.item())
                acc = MF.accuracy(y_hat_x[m], yx[m], task="multiclass",
                                   num_classes=dataset.num_classes)
                Acc.append(acc.item())

        tt = time.time()
        durations.append(tt - t0)
        aveLOSS.append(f"{sum(Loss) / it:.8f}")  
        aveTrain_acc.append(f"{sum(Acc) / it:.8f}")

        model.eval() 
        with torch.no_grad():
           
            val_preds, test_preds = [], []
            val_labels, test_labels = [], []
            for it, sg in enumerate(eval_dataloader):  
                x = sg.ndata["feat"] 
                y = sg.ndata["label"] 
                m_val = sg.ndata["val_mask"].bool()
                m_test = sg.ndata["test_mask"].bool()
                y_hat = model.inference(sg, x)
                val_preds.append(y_hat[m_val])
                val_labels.append(y[m_val])
                test_preds.append(y_hat[m_test])
                test_labels.append(y[m_test])

           
            val_preds = torch.cat(val_preds, 0)  
            val_labels = torch.cat(val_labels, 0)
            test_preds = torch.cat(test_preds, 0)
            test_labels = torch.cat(test_labels, 0)

            val_acc = MF.accuracy(val_preds, val_labels, task="multiclass", num_classes=dataset.num_classes)
            test_acc = MF.accuracy(test_preds, test_labels, task="multiclass", num_classes=dataset.num_classes)
            
            print(f"Val Acc {val_acc.item():.4f} | Test Acc {test_acc.item():.4f}")
            Val_acc.append(val_acc.item())
            Test_acc.append(test_acc.item())


print(f"Average time: {np.mean(durations):.2f}s, std: {np.std(durations):.2f}s")  

with open(f"./exp_result/arx_edgecl.csv", "w",
          newline="") as f: 
    writer = csv.writer(f) 
    writer.writerow(["aveloss", "avetrain_acc", "val_acc", "test_acc", "epoch_time"])  
    for i in range(len(aveLOSS)):  
        writer.writerow(
            [aveLOSS[i], aveTrain_acc[i], Val_acc[i], Test_acc[i], durations[i]])
