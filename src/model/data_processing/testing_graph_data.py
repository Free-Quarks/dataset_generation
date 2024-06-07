import numpy as np
import os, re, time
import pandas as pd
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import GCNConv, GAE, InnerProductDecoder, VGAE
from torch_geometric.utils import train_test_split_edges
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# Directory with graphs data with csv files
DIRECTORY_TO_CSV_FILES = '../../../data/output_csv_graphs'

filename1 = os.path.join(DIRECTORY_TO_CSV_FILES, "output-code-1-graph.csv")
filename2 = os.path.join(DIRECTORY_TO_CSV_FILES, "output-code-2-graph.csv")

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

#df = pd.DataFrame(df)

#print(df['c'].head())

def create_dataframe(csv_string):
    # Using Regex to make columns for node id, labels and properties
    regex = r"<Node id=(\d+) labels=\{\'(.*?)\'\} properties=\{(.*?)\}>"
    match = re.match(regex, csv_string)
    if match:
        node_id, labels, properties = match.groups()
        #print(node_id, labels, properties)
        return node_id, labels, properties
    else:
        return None, None, None

def node_relationship(csv_string):
    # Using Regex to obtain nodes relationshop
    #print(f'csv_string:{csv_string}')
    pattern = r"<Relationship id=(\d+) start_node_id=(\d+) end_node_id=(\d+) nodes=\((.*?)\)"
    match = re.match(pattern, csv_string)
    if match:
        relationship_id, start_node_id, end_node_id, nodes = match.groups()
        return nodes
    else:
        return None


#print(df['c'][0])
# To create a new dataframe, apply the `create_dataframe` function to each row
'''node_id, labels, properties = zip(*df['c'].apply(create_dataframe))
#print(f'node_id:',node_id)
# obtain nodes relationship
nodes_relationship = df['r'].apply(node_relationship)

#print(f'nodes_relationship:',nodes_relationship)
# create a new dataframe
new_df = pd.DataFrame({'node_id': node_id, 'labels': labels, 'properties': properties, 'nodes_relationship': nodes_relationship})

new_df[['start_node_id', 'end_node_id']] = new_df['nodes_relationship'].str.extract('(\d+), (\d+)')

# Convert the string ids to int
new_df['start_node_id'] = new_df['start_node_id'].astype(int)
new_df['end_node_id'] = new_df['end_node_id'].astype(int)

print(new_df.head())
print(new_df['nodes_relationship'])'''

def new_dataframe(filename):
    df = pd.read_csv(filename)
    df = pd.DataFrame(df)

    node_id, labels, properties = zip(*df['c'].apply(create_dataframe))
    # print(f'node_id:',node_id)
    # obtain nodes relationship

    nodes_relationship = df['r'].apply(node_relationship)
    node_id_m, labels_m, properties_m = zip(*df['m'].apply(create_dataframe))

    print(f'nodes_relationship:{nodes_relationship}')

    # print(f'nodes_relationship:',nodes_relationship)
    # create a new dataframe
    new_df = pd.DataFrame(
        {'node_id': node_id, 'labels': labels, 'properties': properties})
    #drop duplicates
    new_df.drop_duplicates(subset=['node_id'], inplace=True)


    new_df_m = pd.DataFrame(
        {'node_id': node_id_m, 'labels': labels_m, 'properties': properties_m})
    new_df_m.drop_duplicates(subset=['node_id'],inplace=True)

    new_df = pd.concat([new_df, new_df_m], ignore_index=True)

    print(f'len(new_df):{len(new_df)}')
    #test_node = new_df['node_id']
    print(f'df.node_id.nunique():{new_df.node_id.nunique()}')

    df_nodes_relationship = pd.DataFrame({'nodes_relationship': nodes_relationship})

    df_nodes_relationship[['start_node_id', 'end_node_id']] = df_nodes_relationship['nodes_relationship'].str.extract('(\d+), (\d+)')
    # Convert the string ids to int
    df_nodes_relationship['start_node_id'] = df_nodes_relationship['start_node_id'].astype(int)
    df_nodes_relationship['end_node_id'] = df_nodes_relationship['end_node_id'].astype(int)



    print(new_df)
    print(df_nodes_relationship)

    return new_df, df_nodes_relationship


def create_graph(df, df_nodes_relationship):
    # Creating a graph
    G = nx.Graph()

    for i, row in df.iterrows():
        node_id = row['node_id']
        labels = row['labels']
        properties = row['properties']

        # Combine labels and properties into a single dictionary
        attributes = {'labels': labels, 'properties': properties}

        G.add_node(node_id, label=labels) #, **properties )
    for i, row in df_nodes_relationship.iterrows():
        start_node_id = row['start_node_id']
        end_node_id = row['end_node_id']
        G.add_edge(start_node_id, end_node_id)
        #G.add_edge(nodes_related)

    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True)
    plt.show()
    return G


df0, df0_nodes_relationship = new_dataframe(filename1)
print(f'df0:{df0}')
G0 = create_graph(df0,df0_nodes_relationship)
print(f'G0:{G0}')
#print(f'G0.number_of_nodes():{G0.number_of_nodes()}')
#df1 = new_dataframe(filename2)
#G1 = create_graph(df1)

'''node2vec0 = Node2Vec(G0, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Node embeddings
model0 = node2vec0.fit(window=10, min_count=1, batch_words=4)
# access embedding -- from `wv` attribute which is word vector
embeddings0 = model0.wv

all_embeddings0 = []
for node in model0.wv.index_to_key:
    embedding = embeddings0[node]
    #print(f"Node: {node}, Embedding: {embedding}")
    all_embeddings0.append(embedding)

node2vec1 = Node2Vec(G1, dimensions=64, walk_length=30, num_walks=200, workers=4)
# Node embeddings
model1 = node2vec1.fit(window=10, min_count=1, batch_words=4)
# access embedding -- from `wv` attribute which is word vector
embeddings1 = model1.wv


all_embeddings1 = []
for node in model1.wv.index_to_key:
    embedding = embeddings1[node]
    #print(f"Node: {node}, Embedding: {embedding}")
    all_embeddings1.append(embedding)



similarity_matrix_cosine_01 = cosine_similarity(all_embeddings0, all_embeddings1)
print(f'similarity_matrix_cosine_01: {similarity_matrix_cosine_01}')
# Use heatmap to plot the similarity matrix for df0 and df1
plt.imshow(similarity_matrix_cosine_01, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Similarity')
plt.title('Cosine Similarity Matrix')
plt.xlabel('Nodes')
plt.ylabel('Nodes')
plt.show()

# Set a similarity threshold
threshold = 0.5
similar = np.sum(similarity_matrix_cosine_01 > threshold)

#Count the number of entries between  0.7 and  1
count = np.sum((similarity_matrix_cosine_01 >  0.5) & (similarity_matrix_cosine_01 <=  1))
print("Number of vectors with cosine similarity between  0.7 and  1:", count)
# Calculate the percentage of similar vectors
total = similarity_matrix_cosine_01.size
print("Total:", total)
percentage_similar = (count / total) * 100
print(f'percentage_similar:{percentage_similar}')

print("---------------------------------")'''
#G.add_edges_from(include_edges)


def data_for_GCN(df, df_nodes_relationship):
    #edge_index = torch.tensor(list(G.edges)).t().contiguous()
    #x = torch.tensor([attributes for _, attributes in G.nodes(data=True)])

    print(df.head())


    # 'labels' are expression, opo, opo, etc...
    # Make use of label encoding to convert string labels to integers
    label_encoder = LabelEncoder()
    new_labels_list = label_encoder.fit_transform(df['labels'].values)
    print(f'new_labels_list:{new_labels_list}')
    print(f'type(new_labels_list):{type(new_labels_list)}')
    print(df['node_id'].astype(int).values)

    #nodes_list =torch.tensor(np.asarray(df['node_id'].astype(int).values))
    nodes_list = df['node_id'].astype(int).values.tolist()
    print(f'nodes_list:{nodes_list}')
    #new_nodes_list = torch.tensor([i for i in range(len(nodes_list))])
    new_nodes_list = [i for i in range(len(nodes_list))]
    print(f'new_nodes_list:{new_nodes_list}')
    new_labels_list = torch.tensor(new_labels_list)

    features_data = {'nodes_list': new_nodes_list, 'new_labels_ist': new_labels_list}
    features_df = pd.DataFrame(features_data)

    x = features_df.to_numpy(dtype=np.float32)
    x = torch.from_numpy(x)

    #print(f'features_df:{features_df}')

    source_nodes = df_nodes_relationship['start_node_id'].values.tolist()
    target_nodes = df_nodes_relationship['end_node_id'].values.tolist()

    print(f'source_nodes:{source_nodes}')

    new_source_nodes = [nodes_list.index(start) for start in source_nodes]
    print(f'new_source_nodes:{new_source_nodes}')
    new_target_nodes = [nodes_list.index(start) for start in target_nodes]
    print(f'new_target_nodes:{new_target_nodes}')

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).t()# to slow

    #combined_array = np.vstack((np.hstack(source_nodes), np.hstack(target_nodes)))
    print(f'edge_index:{edge_index}')
    edge_index = torch.tensor([new_source_nodes, new_target_nodes], dtype=torch.long).t()

    print(f'-------------> newedge_index:{edge_index}')
    # Convert numpy array to a torch tensor
    #edge_index = torch.tensor(combined_array)
    # create data object  to do geometric gnn
    data = Data(x=x, edge_index=edge_index.t().contiguous(),)
                #y=torch.tensor(new_labels_list, dtype=torch.long))
    print(f'data:{data}')
    print(f'data.num_nodes:{data.num_nodes}')
    print(f'data.num_features:{data.num_features}')
    return data


data = data_for_GCN(df0, df0_nodes_relationship)

print(data.edge_index.shape)


'''def create_adjacency_matrix(V, edges):
    adj_matrix = torch.zeros((V, V), dtype=torch.float32)
    print('edges[ 0]:', edges[0])
    print('edges[1]:', edges[1])
    adj_matrix[edges[0], edges[1]] = 1
    adj_matrix[edges[1], edges[0]] = 1

    return adj_matrix


adj_matrix = create_adjacency_matrix(len(data.x), data.edge_index.t())
'''
from torch_geometric.utils import to_dense_adj
adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=len(data.x))[0]


one_in_A = (adj_matrix == 1).any(dim=0)

print(f'adj_matrix: {adj_matrix}')

print(f'one_in_A: {one_in_A}')

'''adjacency_matrix = torch.zeros((len(data.x), len(data.x)))

# Set the adjacency matrix based on the edge indices
for i, j in zip(*data.edge_index):
    adjacency_matrix[i, j] = 1
    adjacency_matrix[j, i] = 1  # Since the adjacency matrix is symmetric

print(f'adjacency_matrix: {adjacency_matrix}')'''


print(f'len(data): {len(data.x)}')
print(f'data.x: {data.x}')
print(f'data.edge_index: {data.edge_index}')
print(f'data.edge_index.t(): {data.edge_index.t()}')
print(f'data.edge_index.t().shape: {data.edge_index.t().shape}')

# has positive edges where the positive edges are in the graph, where d
#transform = RandomLinkSplit(is_undirected=True)

transform = T.Compose([
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0., num_test=0., is_undirected=True,
                      split_labels=True, #disjoint_train_ratio=0.3,
                      add_negative_train_samples=False),#, neg_sampling_ratio=0),
])

#train_data, val_data, test_data = transform(data)
data, _, _ = transform(data)

# parameters
#out_channels = 2
num_features = 2
epochs = 100
print(f'data.num_features:{num_features}')

#model = GAE(GCNEncoder(num_features, out_channels))


#data = train_test_split_edges(data)

#################
# Going through pytorch geometric's official github page on autoencoders
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py -- for our purpose
##################


class VGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden=68):
        super(VGCNEncoder, self).__init__()
        # use Graph Convolutional Network where we have 2 comvolutional NN
        self.conv1 = GCNConv(in_channels, 2 * hidden, cached=True) # here output channels is double of the channels for 2 convolutional NN
        self.conv2_mu = GCNConv(2*hidden, hidden, cached=True)
        self.conv2_logstd = GCNConv(2*hidden, hidden, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv2_mu(x, edge_index)
        log_std = self.conv2_logstd(x, edge_index)
        return mu, log_std



# Call VGAE and apply our encoder
model = VGAE(VGCNEncoder(num_features).to(device), decoder=InnerProductDecoder() )
model = model.to(device) # move model to gpu if available

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f'dir(model):{dir(model)}')
print(f'data:{data}')
#print(f'data.x.shape:{data.shape}')
print(f'data.pos_edge_label_index:{data.pos_edge_label_index}')

print(f'data.edge_index.shape:{data.edge_index.shape}')
print("|||||||||||||||||||||")
print(f'data.x:{data}')
print("---------------")
print(f'data.edge_index:{data.edge_index}')


def train():
    '''
    Train the model
    Returns: loss

    '''
    model.train() # puts model in training mode
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index) #encodes the data
    print(f'z={z}')

    reconstructed = model.decode(z, data.edge_index)
    print(f'reconstructed={reconstructed}')
    loss = model.recon_loss(z, data.pos_edge_label_index)  # compute the reconstructed loss
    print(f'loss={loss}')

    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    kl_loss = model.kl_loss()
    print(f'----> kl_loss={kl_loss}')
    loss.backward()  # backprop
    optimizer.step()  # step on the optimizer
    print(f'----> float(loss): {float(loss)}')
    return float(loss) #z #float(loss)

def evaluate(data):#, neg_edge):
    model.eval() # puts model in evaluation mode
    z = model.encode(data.x, data.edge_index) #encode the data with edges
    #print(f'----> data.pos_edge_label_index={data.pos_edge_label_index}')
    # test data with positive and negative edge label index
    ev = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    print(f'----> ev={ev}')
    return ev



epohchs = 100
times = []
for epoch in range(1, epochs + 1):
    start = time.time()
    z = train()
    '''auc, avg_prec = evaluate(tdata) #, test_data.neg_edge_label_index)
    print(f'Epoch: {epoch} ==> AUC: {auc:.4f}, Avg. Precision: {avg_prec:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")'''
