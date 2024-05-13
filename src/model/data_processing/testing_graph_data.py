import numpy as np
import os, re
import pandas as pd
import torch
from torch_geometric.data import Data
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
        attributes = {'labels': labels}

        G.add_node(node_id, **attributes)
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



#pos = nx.spring_layout(G)

#nx.draw(G, with_labels=True)
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
def data_for_GCN(df, df_nodes_relationship):
    #edge_index = torch.tensor(list(G.edges)).t().contiguous()
    #x = torch.tensor([attributes for _, attributes in G.nodes(data=True)])

    print(df.head())
    nodes_list = np.asarray(df['node_id'].astype(int).values)
    print(f'nodes_list:{nodes_list}')

    # 'labels' are expression, opo, opo, etc...
    # Make use of label encoding to convert string labels to integers
    label_encoder = LabelEncoder()
    new_labels_list = label_encoder.fit_transform(df['labels'].values)

    source_nodes = df_nodes_relationship['start_node_id'].values
    target_nodes = df_nodes_relationship['end_node_id'].values

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    # create data object  to do geometric gnn
    data = Data(x=torch.tensor(nodes_list), edge_index=edge_index,
                y=torch.tensor(new_labels_list, dtype=torch.long))
    print(f'data:{data}')
