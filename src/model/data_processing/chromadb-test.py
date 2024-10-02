import numpy as np
import torch
import chromadb
from chromadb import Documents, EmbeddingFunction
import glob
from io import StringIO
import tqdm
import pandas as pd
from graph_data_with_properties import new_dataframe, graph_positional_encoding
from gae_model_with_prop import GraphAutoEncoder, GAEncoder, GADecoder
from preprocessing_graph_csv import restructure_data
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from transformers import AutoModel, AutoTokenizer

DIRECTORY_TO_CSV_FILES = '../../../data/output_csv_graphs/'

IN_CHANNELS = 7
HIDDDEN_CHANNELS = 40
OUT_CHANNELS = 38
EDGE_DIM = 7
PROPERTIES_DIM = 256
NUM_FEATURES = 13


CHECKPOINT = "Salesforce/codet5p-110m-embedding"
GRAPH_CHECKPOINT = "./models/gae/graph_model.pth"


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

chroma_client = chromadb.Client()


# create a collection (i.e. database) to store embeddings in
#collection = chroma_client.create_collection(name='graph_db')


print("Here")

def datasets(filenames):
    concatenate_dataset = []
    for file in filenames:
        df, df_nodes_relationship = new_dataframe(file)
        df_nodes_relationship = df_nodes_relationship.drop(columns=['nodes_relationship'], axis=1)
        #concatenate_dataset.append(df)
        concatenate_dataset.append(pd.concat([df, df_nodes_relationship], axis=1))
    return concatenate_dataset


def embedding_func(text, model):
    print("text:", text)
    inputs = tokenize.encode(text, return_tensors="pt")
    print("inputs: ", inputs)
    print("inputs.shape: ", inputs.shape)
    outputs = model(inputs)[0]
    print("outputs: ", outputs)
    print("outputs.shape: ", outputs.shape)

    # Convert to list
    embeddings = outputs.tolist()
    print("|||| embeddings: ", embeddings)

    print("---> embeddings: ", embeddings)

    return embeddings

###### TODO: Take df as document.page_content and output a single vector of embedding
###### TODO: Use the embedding vector in
## strings to df to gae model
###### vector db that has the document.pagecontent --- use vectordb
def embed_with_chroma(df, embedding_model):
    embeddings = []
    docs = []
    print('df.shape[0]', df.shape[0])
    print('df.shape[1]', df.shape[1])

    # Process each row in the DataFrame with a progress bar
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        # Create a Document with necessary fields
        document = Document(
            page_content=f"labels: {row['labels']}, properties: {row['properties']}, edge_type: {row['edge_type']}, start_node_id: {str(row['start_node_id'])}, end_node_id: {str(row['end_node_id'])}",  # Text content for embedding
            meta_data={'properties': row['properties']},#, 'edge_type': row['edge_type'], 'start_node_id': str(row['start_node_id']), 'end_node_id': str(row['end_node_id'])},
            id=str(row['node_id'])
        )
        print("document.page_content: ", document.page_content)
        print("type(document.page_content): ", type(document.page_content))
        try:
            doc_embeddings = embedding_func(document.page_content,embedding_model)
            docs.append(document)
            embeddings.append(doc_embeddings)
            #vector_db.add_document(document, doc_embeddings)
        except Exception as e:
            print(f"Failed to embed document: {e}")

    return docs, embeddings

def create_document(df):
    '''

    Args:
        df: Dataframe for a single code file

    Returns: Document for chromadb

    '''
    docs = []
    print('df.shape[0]', df.shape[0])
    print('df.shape[1]', df.shape[1])
    string_df = df.to_string(index=False)
    print("string_df", string_df)
    print("type(string_df)", type(string_df))

    document = Document(page_content=string_df, meta_data=string_df, )
    docs.append(document)

    print("document.page_content: ", document.page_content)
    try:
        doc_embeddings = embedding_func(document.page_content,embedding_model)
            #docs.append(document)
           # embeddings.append(doc_embeddings)
        #vector_db.add_document(document, doc_embeddings)
    except Exception as e:
        print(f"Failed to embed document: {e}")

    return docs


def new_df(documents):

    string_df = documents.page_content
    print("string_df", string_df)

    rows = string_df.split('\n')
    split_lines = []
    for row in rows:
        columns = row.strip().split(maxsplit=len(row.split()))
        split_lines.append(columns)
    print("split_lines:", split_lines)

    #this_df = pd.DataFrame(processed_lines,columns=['node_id', 'labels', 'properties', 'edge_type', 'start_node_id', 'end_node_id'])


    this_df = pd.DataFrame(split_lines[1:], columns=split_lines[0])


    print("this_df", this_df.head())
    print('this_df.shape[0]', this_df.shape[0])
    print('this_df.shape[1]', this_df.shape[1])

    df_1 = pd.DataFrame(this_df[['node_id', 'labels', 'properties']])

    print("df_1", df_1)

    df_2 = this_df[['edge_type','start_node_id', 'end_node_id']]
    df_nodes_relationship = df_2.replace(to_replace='NaN', value=np.nan).dropna()

    print("df_2", df_nodes_relationship)

    return df_1, df_nodes_relationship




class ChromaGraphEmbedding(EmbeddingFunction):
    def __init__(self, codet_checkpoint, graph_checkpoint):
        self.model = GraphAutoEncoder(GAEncoder(PROPERTIES_DIM, EDGE_DIM, NUM_FEATURES, HIDDDEN_CHANNELS, OUT_CHANNELS), GADecoder(EDGE_DIM,IN_CHANNELS, HIDDDEN_CHANNELS, OUT_CHANNELS))#InnerProductDecoder())

        self.model_checkpoint = codet_checkpoint
        self.graph_checkpoint = graph_checkpoint

    def __call__(self, doc) -> chromadb.Embeddings:

        embeddings = []
        graph_model = self.model
        graph_model.load_state_dict(torch.load(self.graph_checkpoint))
        graph_model.eval()


        # print("documents", documents)
        # print("len(document_embeddings)", len(document_embeddings))
        '''graph_list =  []
        for i in range(len(doc)):
            df, df_nodes_relationship = new_df(doc[i])
            restructured_df = restructure_data(df, df_nodes_relationship, self.model_checkpoint)
            print("i , restructured_df:", restructured_df)

            graph_data = graph_positional_encoding([restructured_df])
            print("-->graph_data:", graph_data)

            graph_list.append(graph_data)

            z = graph_model.encode(graph_data.x.to(device), graph_data.x2.to(device), graph_data.edge_index.to(device),
                                 graph_data.edge_attr, graph_data.laplacian_eigenvector_pe.to(device))  # encodes the data

            print("z:", z)
            embeddings.append(z.tolist())'''

        df, df_nodes_relationship = new_df(doc)
        restructured_df = restructure_data(df, df_nodes_relationship, self.model_checkpoint)
        print("i , restructured_df:", restructured_df)

        graph_data = graph_positional_encoding([restructured_df])
        print("-->graph_data:", graph_data)


        z = graph_model.encode(graph_data.x.to(device), graph_data.x2.to(device), graph_data.edge_index.to(device),\
                               graph_data.edge_attr,graph_data.laplacian_eigenvector_pe.to(device))  # encodes the data

        print("z:", z)
        embeddings.append(z.tolist())

        return embeddings


if __name__ == "__main__":
    np.random.seed(5)
    torch.manual_seed(12345)

    filenames = glob.glob(DIRECTORY_TO_CSV_FILES + '*.csv')
    datasets = datasets(filenames)
    print("len(datasets)", len(datasets))
    print("datasets[0]", datasets[0])
    print('datasets[0].shape[0]', datasets[0].shape[0])
    print('datasets[0].shape[1]', datasets[0].shape[1])






    tokenize = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True)

    #vector_db = Chroma(collection_name="graph_sets", embedding_function=embedding_model)

    #client = chromadb.PersistentClient()

    documents = [create_document(datasets[i]) for i in range(len(datasets))]
    print("--------> len(documents)", len(documents))

    #documents = create_document(datasets[0])
    #print("documents", documents)
    #print("len(document_embeddings)", len(document_embeddings))

    #df, df_nodes_relationship = new_df(documents[0])
    #restructured_df = restructure_data(df, df_nodes_relationship,  CHECKPOINT)
    #print("restructured_df:", restructured_df)

    #graph_df = graph_positional_encoding([restructured_df])

    #print("graph_df:", graph_df)




    '''documents_1, document_embeddings_1 = embed_with_chroma(datasets[1], embedding_model)
    print("document_embeddings", document_embeddings)
    print("document_embeddings_1", document_embeddings_1)


    print("len(document_embeddings)", len(document_embeddings))
    print("len(document_embeddings[0])", len(document_embeddings[0]))
    print("len(document_embeddings[1])", len(document_embeddings[1]))
    print("len(document_embeddings[-1])", len(document_embeddings[-1]))

    print("len(document_embeddings_1)", len(document_embeddings_1))



    print("vector_db", vector_db)
    collection = client.get_or_create_collection("graph_data_code")
    print("There are", vector_db.count(), "in the collection")'''


    embedding_function_chroma_graph = ChromaGraphEmbedding(CHECKPOINT, GRAPH_CHECKPOINT)

    persistent_client = chromadb.PersistentClient()  # default settings

    print("persistent_client.list_collections():", persistent_client.list_collections())
    #del persistent_client
    #persistent_client = chromadb.PersistentClient()  # default settings
    # this gets the collection since it's already present
    collection = persistent_client.get_collection("graph_data_code", embedding_function=embedding_function_chroma_graph)
    print("collection", collection)
    #print("There are", collection.count(), "in the collection")
    for i, entry in enumerate(documents):
        print("i", i)
        print("entry", entry)
        collection.add(ids=f"{i}", embeddings=embedding_function_chroma_graph(entry))
        #metadatas=entry.metadata, documents=entry.page_content)
        print(f"{i} of {len(documents)} added to db")

    results = collection.query(
        query_texts=[documents[0].page_content],  # Chroma will embed this for you
        n_results=2  # how many results to return
    )
    print(results['distances'])
    print("End")

