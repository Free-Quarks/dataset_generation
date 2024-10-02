import json
import glob
import numpy as np
import os, time, tqdm
import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv, GAE, InnerProductDecoder, VGAE
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# Directory with graphs data with csv files

from graph_data_with_properties import graph_datasets,graph_positional_encoding
from gae_model_with_prop import GraphAutoEncoder, GAEncoder, GADecoder



DIRECTORY_TO_CSV_FILES = '../../../data/output_csv_graphs/'
np.random.seed(5)
torch.manual_seed(12345)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def evaluate(data):#, neg_edge):
    model.eval() # puts model in evaluation mode
    z = model.encode(data.x, data.edge_index) #encode the data with edges
    #print(f'----> data.pos_edge_label_index={data.pos_edge_label_index}')
    # test data with positive and negative edge label index
    ev = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    print(f'----> ev={ev}')
    return ev







#writer.close()

if __name__ == "__main__":
    np.random.seed(5)
    torch.manual_seed(12345)

    filenames = glob.glob(DIRECTORY_TO_CSV_FILES + '*.csv')
    graph_dataset = graph_datasets(filenames)
    print(f'graph_dataset: {graph_dataset}')
    graph_pe_dataset = graph_positional_encoding(graph_dataset)
    print(f'graph_pe_dataset: {graph_pe_dataset}')
    data = graph_dataset[0]
    new_data = graph_pe_dataset[0]

    adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=len(data.x))[0]

    one_in_A = (adj_matrix == 1).any(dim=0)

    print(f'adj_matrix: {adj_matrix}')

    print(f'one_in_A: {one_in_A}')

    print(f'len(data): {len(data.x)}')
    print(f'data.x: {data.x}')
    print(f'data.edge_index: {data.edge_index}')
    print(f'data.edge_index.t(): {data.edge_index.t()}')
    print(f'data.edge_index.t().shape: {data.edge_index.t().shape}')
    print(f'data.edge_featurese: {data.edge_attrs}')

    # has positive edges where the positive edges are in the graph, where d
    # transform = RandomLinkSplit(is_undirected=True)
    '''data_loader = torch.utils.data.DataLoader(graph_dataset, batch_size=16, shuffle=False)

    transform = T.Compose([
        T.ToDevice(device),
        T.RandomLinkSplit(num_val=0., num_test=0., is_undirected=True,
                          split_labels=True,  # disjoint_train_ratio=0.3,
                          add_negative_train_samples=False),  # , neg_sampling_ratio=0),
    ])'''

    # train_data, val_data, test_data = transform(data)
    #graph_dataset, _, _ = transform(graph_dataset)
    #graph_pe_dataset, _, _ = transform(graph_pe_dataset)
    # parameters
    # out_channels = 2
    #print()
    num_features = data.num_features

    TOKENIZERS_PARALLELISM = False


    epochs = 100
    print(f'data.num_features:{num_features}')

    # runs/vgae_with_laplacian_pe k =10
    # writer = SummaryWriter('runs/vgae_hidden=24_with_laplacian_pe_k=10') # Using tensorboard

    model = GraphAutoEncoder(GAEncoder(256, 7, num_features, 40, 38), GADecoder(7,38, 40, 38))#InnerProductDecoder())
    #model = GraphAutoEncoder(GAEncoder(num_features, 24, 12), GADecoder())
    model = model.to(device)  # move model to gpu if available
    print(model)

    # Initialize the optim izer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f'dir(model):{dir(model)}')
    print(f'data:{data}')
    # print(f'data.x.shape:{data.shape}')
    #print(f'data.pos_edge_label_index:{data.pos_edge_label_index}')
    # print(f'laplacian_encoded_graph_data.pos_edge_label_index:{laplacian_encoded_graph_data.pos_edge_label_index}')

    print(f'data.edge_index.shape:{data.edge_index.shape}')
    print("|||||||||||||||||||||")
    print(f'data.x:{data}')
    print("---------------")
    print(f'data.edge_index:{data.edge_index}')
    print(f'data.edge_attrs:{data.edge_attr}')

    epochs = 100

    loss_func = torch.nn.MSELoss()
    times = []
    for epoch in range(epochs):
        start = time.time()

        for i, pe_data in tqdm.tqdm(enumerate(graph_pe_dataset)):
            print("i", i)
            num_features = pe_data.num_features
            transform = T.Compose([
                T.ToDevice(device),
                T.RandomLinkSplit(num_val=0., num_test=0., is_undirected=True,
                                  split_labels=True,  # disjoint_train_ratio=0.3,
                                  add_negative_train_samples=False),  # , neg_sampling_ratio=0),
            ])

            pe_data, _, _ = transform(pe_data)
            gdata, _, _ = transform(graph_dataset[i])

            model.train()
            optimizer.zero_grad()
            num_nodes = pe_data.x.shape[0]
            print(f'num_nodes:{num_nodes}')
            print(f'pe_data:{pe_data}')
            print(f'pe_data.x.shape:{pe_data.x.shape}')
            print(f'pe_data.x2.shape:{pe_data.x2.shape}')
            z = model.encode(pe_data.x.to(device), pe_data.x2.to(device), pe_data.edge_index.to(device), pe_data.edge_attr , pe_data.laplacian_eigenvector_pe.to(device))  # encodes the data
            print(f'z={z}')
            #reconstructed = model.decode(z.to(device), num_nodes, pe_data.edge_index.to(device))
            reconstructed = model.decode(z.to(device), num_nodes, pe_data.edge_index.to(device), pe_data.edge_attr.to(device), pe_data.laplacian_eigenvector_pe.to(device))
            print(f'reconstructed={reconstructed}')
            print(f'graph_dataset[i]:{graph_dataset[i]}')
            print(f'pe_data:{pe_data}')
            #gdata = graph_dataset[i]
            print(f'graph_dataset[i].pos_edge_label_index:{gdata.pos_edge_label_index}')
            loss = loss_func(pe_data.x.to(device).to(device), reconstructed.to(device))
            print(f'loss={loss}')
            loss.backward()  # backprop
            optimizer.step()  # step on the optimizer
            print(f'----> float(loss): {float(loss)}')

        # writer.add_scalar('Loss/train', loss.item(), epoch)

        print(model)

        torch.save(model.state_dict(), './models/gae/graph_model.pth')



        '''auc, avg_prec = evaluate(tdata) #, test_data.neg_edge_label_index)
        print(f'Epoch: {epoch} ==> AUC: {auc:.4f}, Avg. Precision: {avg_prec:.4f}')
        times.append(time.time() - start)
        print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")'''
        print(times.append(time.time() - start))