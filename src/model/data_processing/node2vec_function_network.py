import numpy as np
import pandas as pd
import torch
import os, re
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from networkx.drawing.nx_pydot import graphviz_layout

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
    #filename0 = os.path.join(function_net_txt_files_dir, "output-code-0.txt")
    with open(filename, 'r') as f:
        lines = f.readlines()

    fn_array_dict = {}
    for line in lines:
        elements = line.split()
        node = elements[0]
        fn_array_dict[node] = elements[1:]
    #print(fn_array_dict)

    df = pd.DataFrame(list(fn_array_dict.items()), columns=['Node', 'Labels'])
    #print(df)
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
    pos = graphviz_layout(G, prog="dot")
    #nx.draw(G,pos,with_labels=True)
    nx.draw(G, pos)
    #plt.show()
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
#print(f"all_embeddings0[0].shape:", all_embeddings0[0].shape)
#print(f"np.array(all_embeddings0).shape:", np.array(all_embeddings0).shape)
#print("================================")



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
#print(f"all_embeddings1[0].shape:", all_embeddings1[0].shape)
#print(f"np.array(all_embeddings1).shape:", np.array(all_embeddings1).shape)



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

#Count the number of entries between  0.7 and  1
count = np.sum((similarity_matrix_cosine_01 >  0.5) & (similarity_matrix_cosine_01 <=  1))
print("Number of vectors with cosine similarity between  0.7 and  1:", count)
# Calculate the percentage of similar vectors
total = similarity_matrix_cosine_01.size
print("Total:", total)
percentage_similar = (count / total) * 100
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


# Calculate the depth of a node using Depth First Search
def depth_first_search(G, node, parent=None, depth=0):
    depths = [depth]  # store depths of all child nodes
    for neighbor in G.neighbors(node):
        if neighbor != parent:
            depths.extend(depth_first_search(G, neighbor, node, depth+1))
    return depths

# Calculate the sum of depths for all nodes
def calculate_sum_depth(G):
    depths = []
    total_sum = 0
    for node in G:
        depth = depth_first_search(G0, node)
        depths.append(sum(depth))
        total_sum += sum(depth_first_search(G0, node))
    return total_sum, max(depths), min(depths)

total_sum, max_depths, min_depths = calculate_sum_depth(G0)
print("------")
print(total_sum, max_depths, min_depths)


# nx.average_shortest_path_length() assumes that the graph is undirected.
# So for directed graph, compute the average shortest path length for subgraphs
def shortest_path_length(G):
    path_length = []
    for C in nx.connected_components(G):
        subgraph = G.subgraph(C).copy()
        #print(nx.average_shortest_path_length(subgraph))
        path_length.append(nx.average_shortest_path_length(subgraph))
    #print(np.average(path_length),np.max(path_length), np.min(path_length))
    return np.average(path_length),np.max(path_length), np.min(path_length)

avg_path_length, max_path_length, min_path_length = shortest_path_length(G0)
print("------")
print(f'avg_path_length:{avg_path_length}, max_path_length:{max_path_length}, min_path_length:{min_path_length}')
print("------")

depths = []
total_sum =  0
for node in G0:
    depth = depth_first_search(G0, node)
    depths.append(sum(depth))
    total_sum += sum(depth_first_search(G0, node))
print("G0.nodes():",G0.nodes())
print("depths:", depths)
print("len(depths):", len(depths))
print("sum(depths):", sum(depths))
print("The sum of depths of all subtree nodes is:", total_sum)
print("----")
depths = []
total_sum =  0
for node in G1:
    depth = depth_first_search(G1, node)
    depths.append(sum(depth))
    total_sum += sum(depth_first_search(G1, node))
print("G1.nodes():",G1.nodes())
print("depths:", depths)
print("len(depths):", len(depths))
print("sum(depths):", sum(depths))
print("The sum of depths of all subtree nodes is:", total_sum)




#avg_shortest_path_length = nx.average_shortest_path_length(G0)
#print("avg_shortest_path_length:",avg_shortest_path_length)


#graph_edit_distance = nx.graph_edit_distance(G0, G1)
#print("The graph edit distance between G0 and G2 is:", graph_edit_distance)


# Define a custom sorting key that extracts the numerical part of the filename
def sorting_file(filename):
    '''
    Sorts the file in order
    '''
    match = re.match(r'output-code-(\d+)\.txt', filename)
    if match:
        return int(match.group(1))
    else:
        return 0

# Get the list of files sorted
files = sorted(os.listdir(function_net_txt_files_dir), key=sorting_file)
#for i, file in enumerate(files):
    #print(i, file)
results_cosine_similarity = {}
similarity_percent = []
with open('cos_similarity.txt', 'w') as f, open('depth.txt', 'w') as f2, open('path_length.txt', 'w') as f3:
    # Loop through each file
    #for i in range(10):
    for i, file in enumerate(files):
        #print(i, files[i])
        path = os.path.join(function_net_txt_files_dir, files[i])
        df0 = dataframe(path)
        G0 = graph(df0)
        total_sum_G0, max_depths_G0, min_depths_G0 = calculate_sum_depth(G0)
        print(f"{files[i]}: total_sum={total_sum_G0}, max_depths={max_depths_G0}, min_depths={min_depths_G0}", file=f2)
        avg_path_length, max_path_length, min_path_length = shortest_path_length(G0)
        print(f"{files[i]}: avg_path_length={avg_path_length}, max_path_length={max_path_length}, min_path_length={min_path_length}", file=f3)

        # Precompute probabilities and generate random walks
        node2vec0 = Node2Vec(G0, dimensions=6, walk_length=2, num_walks=3, workers=4)
        # Node embeddings
        model0 = node2vec0.fit(window=4, min_count=1, batch_words=4)
        # access embedding -- from `wv` attribute which is word vector
        embeddings0 = model0.wv
        all_embeddings0 = []
        for node in model0.wv.index_to_key:
            embedding = embeddings0[node]
            # print(f"Node: {node}, Embedding: {embedding}")
            all_embeddings0.append(embedding)


        # Compare the current file with all other files
        for j in range(i + 1, len(files)):
            #print(files[j])
            # Read the content of the other file into another vector
            #with open(os.path.join(function_net_txt_files_dir, files[j]), 'r') as f:
            pathj = os.path.join(function_net_txt_files_dir, files[j])
            df1 = dataframe(pathj)
            G1 = graph(df1)
            #total_sum_G1, max_depths_G1, min_depths_G1 = calculate_sum_depth(G1)
            #print(f"{files[j]}: total_sum={total_sum_G1}, max_depths={max_depths_G1}, min_depths={min_depths_G1}", file=f2)
            node2vec1 = Node2Vec(G1, dimensions=6, walk_length=2, num_walks=3, workers=4)
            # Node embeddings
            model1 = node2vec1.fit(window=4, min_count=1, batch_words=4)
            # access embedding -- from `wv` attribute which is word vector
            embeddings1 = model1.wv

            all_embeddings1 = []
            for node in model1.wv.index_to_key:
                embedding = embeddings1[node]
                # print(f"Node: {node}, Embedding: {embedding}")
                all_embeddings1.append(embedding)

            #cosine similarity
            similarity_matrix_cosine_01 = cosine_similarity(all_embeddings0, all_embeddings1)

            count = np.sum((similarity_matrix_cosine_01 > 0.5) & (similarity_matrix_cosine_01 <= 1))
            #print("Number of vectors with cosine similarity between  0.5 and  1:", count)
            # Calculate the percentage of similar vectors
            total = similarity_matrix_cosine_01.size
            #print("Total:", total)
            percentage_similar = (count / total) * 100
            similarity_percent.append(percentage_similar)
            # Store the result in the dictionary
            key = f"{files[i]} vs {files[j]}"
            #print(key)
            value= f"{count}/{total};Percent similarity is {percentage_similar}"
            print(f"{key}: {value}", file=f)
            results_cosine_similarity[key] = value
    print("---------------------")
    print(f"list of cosine similarity: {similarity_percent}", file=f)
    print(f"Minimum of cos similarity: {min(similarity_percent)}", file=f)
    print(f"Maximum of cos similarity: {max(similarity_percent)}", file=f)
    print(f"Average of cos similarity: {np.average(similarity_percent)}", file=f)





        #print(f'results_cosine_similarity:',results_cosine_similarity)
# Initialize an empty dictionary to store the results
