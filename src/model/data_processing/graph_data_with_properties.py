import glob
import numpy as np
import re
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import networkx as nx
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from networkx.drawing.nx_pydot import graphviz_layout

# Directory with graphs data with csv files
DIRECTORY_TO_CSV_FILES = '../../../data/output_csv_graphs/'
checkpoint = "Salesforce/codet5p-110m-embedding"

tokenize = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)

text = "name"
tokens = tokenize.encode(text, return_tensors="pt")
embeddings = model(tokens)
print(f'embeddings:{embeddings}')
print(f'embeddings.shape:{embeddings.shape}')
# Set device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def create_dataframe(row):
    '''
    From the csv files, obtain the nodes, labels and it's properties using search patterns
    from each row of the csv file
    Args:
        row: row of csv file

    Returns: node_id, labels, properties

    '''

    print(row)
    regex1 = r"<Node id=(\d+) labels=\{\'(.*?)\'\} properties=\{\'name\': \'(.*?)\'\,"
    regex2 = r"<Node id=(\d+) labels=\{\'(.*?)\'\} properties=\{\'index\': \'(.*?)\'\,"
    regex3 = r"<Node id=(\d+) labels=\{\'(.*?)\'\} properties=\{\'gromet(.*?)\,"
    match1 = re.search(regex1, row)
    match2 = re.search(regex2, row)
    match3 = re.search(regex3, row)

    if match1:
        node_id, labels, properties = match1.groups()
        print(node_id, labels, properties)
        return node_id, labels, properties
    elif match2:
        node_id, labels, properties = match2.groups()
        properties = "index"
        print(node_id, labels, properties)
        return node_id, labels, properties
    elif match3:
        node_id, labels, properties = match3.groups()
        properties = "gromet-version"
        print(node_id, labels, properties)
        return node_id, labels, properties
    else:
        #print(node_id, labels, properties)
        return 'None', 'None', 'None'

def node_relationship(row):
    '''
    Finds which nodes are related/connected
    Args:
        row: row of the csv files

    Returns nodes
    '''
    # "<Relationship id=12321 start_node_id=7834 end_node_id=7928 nodes=(7834, 7928) type=Contains properties={}>"
    pattern = r"<Relationship id=(\d+) start_node_id=(\d+) end_node_id=(\d+) nodes=\((.*?)\) type=(.*?) "
    match = re.match(pattern, row)
    if match:
        relationship_id, start_node_id, end_node_id, nodes, edge_type = match.groups()
        #print(nodes)
        #print(f'edge_type:{edge_type}')
        return nodes, edge_type
    else:
        return 'None', 'None'

def new_dataframe(filename):
    df = pd.read_csv(filename)
    df = pd.DataFrame(df)

    node_id, labels, properties = zip(*df['c'].apply(create_dataframe))
    # print(f'node_id:',node_id)
    # obtain nodes relationship
    print("======================================================================================")

    nodes_relationship, edge_type = zip(*df['r'].apply(node_relationship))
    node_id_m, labels_m, properties_m = zip(*df['m'].apply(create_dataframe))
    print("node_id_m, labels_m, properties_m", node_id_m, labels_m, properties_m)
    #print(f'nodes_relationship:{nodes_relationship}')
    #print(f'edge_type:{edge_type}')

    # print(f'nodes_relationship:',nodes_relationship)
    # create a new dataframe
    new_df = pd.DataFrame(
        {'node_id': node_id, 'labels': labels, 'properties': properties})
    #drop duplicates
    new_df.drop_duplicates(subset=['node_id'], inplace=True)
    print(f'new_df: {new_df}')

    new_df_m = pd.DataFrame(
        {'node_id': node_id_m, 'labels': labels_m, 'properties': properties_m})
    new_df_m.drop_duplicates(subset=['node_id'],inplace=True)

    new_df = pd.concat([new_df, new_df_m], ignore_index=True)
    new_df = new_df.replace(to_replace='None', value=np.nan).dropna()

    print(f'len(new_df):{len(new_df)}')
    #test_node = new_df['node_id']
    print(f'df.node_id.nunique():{new_df.node_id.nunique()}')

    df_nodes_relationship = pd.DataFrame({'nodes_relationship': nodes_relationship, 'edge_type': edge_type})

    df_nodes_relationship[['start_node_id', 'end_node_id']] = df_nodes_relationship['nodes_relationship'].str.extract('(\d+), (\d+)')
    # Convert the string ids to int
    df_nodes_relationship['start_node_id'] = df_nodes_relationship['start_node_id'].astype(int)
    df_nodes_relationship['end_node_id'] = df_nodes_relationship['end_node_id'].astype(int)



    #df_nodes_relationship['edge_type'] = df_nodes_relationship['edge_type']



    print(f'new_df= {new_df}')
    print("df_nodes_relationship", df_nodes_relationship)

    return new_df, df_nodes_relationship

def create_graph(df, df_nodes_relationship):
    print(df)
    # Creating a graph
    G = nx.Graph()

    for i, row in df.iterrows():
        print(f'i:{i}')
        node_id = row['node_id']
        labels = row['labels']
        properties = row['properties']

        print(f'node_id={node_id}, labels={labels}, properties={properties}')
        # Combine labels and properties into a single dictionary
        attributes = {'labels': labels, 'properties': properties}

        #if node_id is not None:
        G.add_node(node_id, label=labels) #, **properties )
        #else:
            #G.add_node(node_id, label=labels) #, **properties )

    for i, row in df_nodes_relationship.iterrows():
        start_node_id = row['start_node_id']
        end_node_id = row['end_node_id']
        G.add_edge(start_node_id, end_node_id)
        #G.add_edge(nodes_related)

    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True)
    plt.show()
    return G


def encode_text_and_generate_embeddings(df):

    properties_encoded_list = []

    for i, row in df.iterrows():
        inputs = tokenize.encode(
            df.iloc[i]['properties'], return_tensors="pt")

        # Generate embeddings
        outputs = model(inputs)
        outputs = torch.flatten(outputs)
        properties_encoded_list.append(outputs.detach().numpy()) #.detach().numpy())

    print(f'new_list_len', len(properties_encoded_list))
    print(f'new_list[0]', len(properties_encoded_list[0]))
    print(f'new_list', properties_encoded_list)

    #df_encoded_text = pd.DataFrame(torch.tensor(properties_encoded_list), columns=properties_encoded_list)

    #print(f'df_encoded_text:{df_encoded_text}')

    properties_encoded_list = torch.tensor(np.array(properties_encoded_list))
    print(f'properties_encoded_list.shape', properties_encoded_list.shape)


    return properties_encoded_list


def data_for_GCN(df, df_nodes_relationship):

    unique_labels = df['labels'].unique()
    #print(f'unique_labels: {unique_labels}')


    label_mapping = {'Function': 1, 'Expression': 2, 'Opo': 3, 'For_Loop': 4, 'Primitive': 5, 'Pil': 6, 'Predicate': 7, 'Opi': 8, 'Abstract': 9, 'Literal': 10, 'Metadata': 11, 'Pol': 12, 'Module': 13}
    #print("len(label_mapping)", len(label_mapping))
    df['labels'] = df['labels'].map(label_mapping)

    df['labels'] = df['labels'].tolist()
    #print(f"df[labels] after mapping: {np.array(df['labels'])}")

    mapped_labels = np.array(df['labels'].tolist())

    print(f'mapped_labels:{mapped_labels}')

    encoded_vectors = []
    for label in mapped_labels:
        # Create a binary vector of length equal to total_features
        binary_vector = np.zeros(len(label_mapping), dtype=int)
        # Set the bit corresponding to the label's feature to 1
        binary_vector[label - 1] = 1  # Assuming labels start from 1
        encoded_vectors.append(binary_vector)

    #print(f'encoded_vectors:{encoded_vectors}')
    #print(f'torch.tensor(encoded_vectors).shape:{torch.tensor(encoded_vectors).shape}')

    edge_type_mapping = {'Metadata': 1, 'Port_Of': 2, 'Contains': 3, 'Wire': 4, 'Pre': 5, 'Condition': 5, 'Body': 6}
    df_nodes_relationship['edge_type'] = df_nodes_relationship['edge_type'].map(edge_type_mapping)

    df_nodes_relationship['edge_type'] = df_nodes_relationship['edge_type'].tolist()

    mapped_edge_type = np.array(df_nodes_relationship['edge_type'].tolist())

    #print(f'mapped_edge_type:{mapped_edge_type}')

    encoded_edge_type_vectors = []
    for type in mapped_edge_type:
        vector = np.zeros(len(edge_type_mapping), dtype=int)
        vector[type - 1] = 1
        encoded_edge_type_vectors.append(vector)

    print(f'encoded_edge_type_vectors:{encoded_edge_type_vectors}')



    '''print(len(label_mapping))
    #initialize
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded_vectors = one_hot_encoder.fit_transform(mapped_labels.reshape(-1, 1)) #.reshape(len(label_mapping), 1))
    print(f'one_hot_encoded_vectors: {one_hot_encoded_vectors}')'''


    # Convert vectors to df
    one_hot_encoded_df = pd.DataFrame(torch.tensor(encoded_vectors), columns=encoded_vectors)
    one_hot_encoded_df = one_hot_encoded_df.to_numpy(dtype=np.float32)
    print(f'one_hot_encoded_df:{one_hot_encoded_df}')

    # Convert vectors to df
    one_hot_encoded_edge_attr_df = pd.DataFrame(torch.tensor(encoded_edge_type_vectors), columns=encoded_edge_type_vectors)
    one_hot_encoded_edge_attr_df = one_hot_encoded_edge_attr_df.to_numpy(dtype=np.float32)
    print(f'one_hot_encoded_edge_attr_df:{one_hot_encoded_edge_attr_df}')

    encoded_property_tensor = encode_text_and_generate_embeddings(df)

    print(f'encoded_property_tensor:{encoded_property_tensor}')


    x = torch.tensor(one_hot_encoded_df)



    edge_features = torch.tensor(one_hot_encoded_edge_attr_df)
    #x = torch.from_numpy(x)
    print("one_hot_encoded_df.values", x)



    label_encoder = LabelEncoder()
    new_labels_list = label_encoder.fit_transform(df['labels'].values)
    print(f'df[labels]:{new_labels_list}')
    print(f'new_labels_list:{new_labels_list}')
    print("df.head()", df.head())
    #print(df['node_id'].astype(int).values)

    #nodes_list =torch.tensor(np.asarray(df['node_id'].astype(int).values))
    nodes_list = df['node_id'].astype(int).values.tolist()
    print(f'nodes_list:{nodes_list}')
    #new_nodes_list = torch.tensor([i for i in range(len(nodes_list))])
    new_nodes_list = [i for i in range(len(nodes_list))]
    print(f'new_nodes_list:{new_nodes_list}')
    new_labels_list = torch.tensor(new_labels_list)

    features_data = {'nodes_list': new_nodes_list, 'new_labels_ist': new_labels_list}
    features_df = pd.DataFrame(features_data)
    print(f'features_df:{features_df}')

    #x = features_df.to_numpy(dtype=np.float32)
    #x = torch.from_numpy(x)

    #print(f'features_df:{features_df}')

    source_nodes = df_nodes_relationship['start_node_id'].values.tolist()
    target_nodes = df_nodes_relationship['end_node_id'].values.tolist()

    print(f'source_nodes:{source_nodes}')
    print(f'nodes_list:{nodes_list}')

    new_source_nodes = [nodes_list.index(start) for start in source_nodes]
    print(f'new_source_nodes:{new_source_nodes}')
    print(f'target_nodes:{target_nodes}')
    new_target_nodes = [nodes_list.index(targ) for targ in target_nodes]
    print(f'new_target_nodes:{new_target_nodes}')

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long).t()# to slow

    #combined_array = np.vstack((np.hstack(source_nodes), np.hstack(target_nodes)))
    print(f'edge_index:{edge_index}')
    edge_index = torch.tensor([new_source_nodes, new_target_nodes], dtype=torch.long).t()

    print(f'-------------> newedge_index:{edge_index}')

    x2 = encoded_property_tensor
    # Convert numpy array to a torch tensor
    #edge_index = torch.tensor(combined_array)
    # create data object  to do geometric gnn
    data = Data(x=x, x2=x2, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
                #y=torch.tensor(new_labels_list, dtype=torch.long))

    print(f'data.num_nodes:{data.num_nodes}')
    print(f'data.num_features:{data.num_features}')
    return data

def graph_datasets(filenames):
    graph_dataset = []
    for file in filenames:
        df, df_nodes_relationship = new_dataframe(file)
        graph_data = data_for_GCN(df, df_nodes_relationship)
        graph_dataset.append(graph_data)
    return graph_dataset

def graph_positional_encoding(graph_dataset):
    graph_pe_dataset = []
    for graph in graph_dataset:
        k = graph.num_features
        laplacian_transform = AddLaplacianEigenvectorPE(k=k, is_undirected=True)
        laplacian_encoded_graph_data = laplacian_transform(graph)
        graph_pe_dataset.append(laplacian_encoded_graph_data)
    return graph_pe_dataset

if __name__ == "__main__":
    filenames = glob.glob(DIRECTORY_TO_CSV_FILES + '*.csv')
    graph_dataset = graph_datasets(filenames)
    print(f'graph_dataset: {graph_dataset}')
    graph_pe_dataset = graph_positional_encoding(graph_dataset)
    print(f'graph_pe_dataset: {graph_pe_dataset}')
