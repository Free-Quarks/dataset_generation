import numpy as np
import re
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer
from torch_geometric.transforms import AddLaplacianEigenvectorPE

CHECKPOINT = "Salesforce/codet5p-110m-embedding"


def encode_text_and_generate_embeddings(df):
    tokenize = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True)

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

def restructure_data(df, df_nodes_relationship, CHECKPOINT):

    unique_labels = df['labels'].unique()
    print(f'unique_labels: {unique_labels}')


    label_mapping = {'Function': 1, 'Expression': 2, 'Opo': 3, 'For_Loop': 4, 'Primitive': 5, 'Pil': 6, 'Predicate': 7, 'Opi': 8, 'Abstract': 9, 'Literal': 10, 'Metadata': 11, 'Pol': 12, 'Module': 13}
    #print("len(label_mapping)", len(label_mapping))
    df['labels'] = df['labels'].map(label_mapping)

    df['labels'] = df['labels'].tolist()
    print(f"df[labels] after mapping: {np.array(df['labels'])}")

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

    print(f'mapped_edge_type:{mapped_edge_type}')

    encoded_edge_type_vectors = []
    for type in mapped_edge_type:
        print("type:",type)
        vector = np.zeros(len(edge_type_mapping), dtype=int)
        vector[int(type) - 1] = 1
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

    source_nodes = df_nodes_relationship['start_node_id'].astype(float).astype(int).values.tolist()
    target_nodes = df_nodes_relationship['end_node_id'].astype(float).astype(int).values.tolist()

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