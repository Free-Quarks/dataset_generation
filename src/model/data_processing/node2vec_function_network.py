import numpy as np
import pandas as pd
import torch
import os
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Directory with tokenized funtion networks .txt files
function_net_txt_files_dir = '../../../data/function_nets_tokenized'

filename0 = os.path.join(function_net_txt_files_dir, "output-code-0.txt")
#filename = os.path.join(function_net_txt_files_dir, "test.txt")
filename1 = os.path.join(function_net_txt_files_dir, "output-code-1.txt")

'''with open(filename, 'r') as f:
    lines = f.readlines()

fn_array_dict = {}
for line in lines:
    elements = line.split()
    node = elements[0]
    fn_array_dict[node] = elements[1:]
print(fn_array_dict)

df = pd.DataFrame(list(fn_array_dict.items()), columns=['Node', 'Labels'])
print(df)

print(df.iloc[0])'''

def dataframe(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    fn_array_dict = {}
    for line in lines:
        elements = line.split()
        node = elements[0]
        fn_array_dict[node] = elements[1:]
    print(fn_array_dict)

    df = pd.DataFrame(list(fn_array_dict.items()), columns=['Node', 'Labels'])
    print(df)
    return df


#G = nx.Graph()

def iterate_labels(node, df, G):
    # Figure out the row that cpnnects to the node in the df
    row = df[df['Node'] == node]
    # Figure out which nodes are connected
    nodes_connected = row['Labels'].tolist()[0] if len(row['Labels'].tolist()) > 0 else []

    # loop around all the nodes that are connected together
    for connect in nodes_connected:
        # Add edge between "node" and "connect"
        G.add_edge(node, connect)
        # Include Recursively call to this function for the connected nodes
        iterate_labels(connect, df, G)


# Call the function for each node in the dataframe
'''for _, row in df.iterrows():
    node = row['Node']
    iterate_labels(node)

nx.draw(G, with_labels=True)'''


def graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        node = row['Node']
        iterate_labels(node, df, G)
    nx.draw(G, with_labels=True)

    return G


# Show the plot
#plt.show()

df0 = dataframe(filename0)
df1 = dataframe(filename1)
G0 = graph(df0)
G1 = graph(df1)

# Using python's Node2Vec object from `node2vec algorithm`
# Precompute probabilities and generate random walks
node2vec0 = Node2Vec(G0, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Node embeddings
model0 = node2vec0.fit(window=10, min_count=1, batch_words=4)
# access embedding -- from `wv` attribute which is word vector
embeddings0 = model0.wv
#print(f"embeddings: {embeddings}")
#print(f"embeddings: {embeddings}")

#embedding = embeddings['FUNCTION1']
#print(f"embedding: {embedding}")

#embedding2 = embeddings['EXPRESSION1']
#print(f"embedding: {embedding2}")

all_embeddings0 = []
for node in model0.wv.index_to_key:
    embedding = embeddings0[node]
    #print(f"Node: {node}, Embedding: {embedding}")
    all_embeddings0.append(embedding)
#print(f"all_embeddings0: {all_embeddings0}")
#print(f"all_embeddings.shape:", len(all_embeddings))
print(f"all_embeddings0[0].shape:", all_embeddings0[0].shape)
print(f"np.array(all_embeddings0).shape:", np.array(all_embeddings0).shape)
print("================================")



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
#print(f"all_embeddings0: {all_embeddings1}")
#print(f"all_embeddings.shape:", len(all_embeddings))
print(f"all_embeddings1[0].shape:", all_embeddings1[0].shape)
print(f"np.array(all_embeddings1).shape:", np.array(all_embeddings1).shape)



##https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise



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

# Calculate the percentage of similar vectors
total = similarity_matrix_cosine_01.size
percentage_similar = (similar / total) * 100
print(f'percentage_similar:{percentage_similar}')

print("---------------------------------")

similarity_matrix_euclidean_01 = euclidean_distances(all_embeddings0, all_embeddings1)
print(f'similarity_matrix_euclidean_01: {similarity_matrix_euclidean_01}')
# Use heatmap to plot the similarity matrix for df0 and df1
plt.imshow(similarity_matrix_euclidean_01, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Similarity')
plt.title('Euclidean Similarity Matrix')
plt.xlabel('Nodes')
plt.ylabel('Nodes')
plt.show()

# Set a similarity threshold
euclidean_threshold = 0.5
euclidean_similar = np.sum(similarity_matrix_euclidean_01 > euclidean_threshold)

# Calculate the percentage of similar vectors
euclidean_total = similarity_matrix_cosine_01.size
percentage_euclidean_similar = (euclidean_similar / euclidean_total) * 100
print(f'percentage_euclidean_similar: {percentage_euclidean_similar}')