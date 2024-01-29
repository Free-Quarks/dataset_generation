import numpy as np
import pandas as pd
import torch
import os
import networkx as nx
import matplotlib.pyplot as plt
from node2vec import Node2Vec

# Directory with tokenized funtion networks .txt files
function_net_txt_files_dir = '../../../data/function_nets_tokenized'

#filename = os.path.join(function_net_txt_files_dir, "output-code-0.txt")
filename = os.path.join(function_net_txt_files_dir, "test.txt")


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

print(df.iloc[0])


G = nx.Graph()

'''for indx, row in df.iterrows():
    if indx> 0: # Skip the first row because it doesn't represent a function
        function = row['Node']
        print("function:", function)
        components = row['Labels']
        print("components:", components)
        for component in components:
            G.add_edge(function, component)
'''


def iterate_labels(node):
    # Figure out the row that cpnnects to the node in the df
    row = df[df['Node'] == node]
    # Figure out which nodes are connected
    nodes_connected = row['Labels'].tolist()[0] if len(row['Labels'].tolist()) > 0 else []

    # loop around all the nodes that are connected together
    for connect in nodes_connected:
        # Add edge between "node" and "connect"
        G.add_edge(node, connect)
        # Include Recursively call to this function for the connected nodes
        iterate_labels(connect)


# Call the function for each node in the dataframe
for _, row in df.iterrows():
    node = row['Node']
    iterate_labels(node)

#from mpl_interactions import ioff, panhandler, zoom_factory
#%matplotlib widget

#with plt.ioff():
#    figure, axis = plt.subplots()
nx.draw(G, with_labels=True)
#disconnect_zoom = zoom_factory(axis)
# Enable scrolling and panning with the help of MPL
# Interactions library function like panhandler.
#pan_handler = panhandler(figure)
#display(figure.canvas)
# Show the plot
plt.show()

# Using python's Node2Vec object from `node2vec algorithm`
# Precompute probabilities and generate random walks
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Node embeddings
model = node2vec.fit(window=10, min_count=1, batch_words=4)
# access embedding -- from `wv` attribute which is word vector
embeddings = model.wv

embedding = embeddings['FUNCTION1']
print(f"embedding: {embedding}")

embedding2 = embeddings['EXPRESSION1']
print(f"embedding: {embedding2}")

for node in model.wv.index_to_key:
    embedding = embeddings[node]
    print(f"Node: {node}, Embedding: {embedding}")
