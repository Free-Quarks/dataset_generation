from typing import List

import numpy as np
import torch
import chromadb
from chromadb import Documents, Embeddings # EmbeddingFunction
import glob
from io import StringIO
import tqdm
import pandas as pd
from pandas import DataFrame

from graph_data_with_properties import new_dataframe, graph_positional_encoding
from gae_model_with_prop import GraphAutoEncoder, GAEncoder, GADecoder
from preprocessing_graph_csv import restructure_data
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from typing import Optional, Union, TypeVar, List, Dict, Any, Tuple, cast
from numpy.typing import NDArray
import numpy as np
from typing_extensions import TypedDict, Protocol, runtime_checkable
from enum import Enum
from pydantic import Field
import chromadb.errors as errors
from chromadb.types import (
    Metadata,
    UpdateMetadata,
    Vector,
    #PyVector,
    LiteralValue,
    LogicalOperator,
    WhereOperator,
    OperatorExpression,
    Where,
    WhereDocumentOperator,
    WhereDocument,
)
from inspect import signature
from tenacity import retry

# Re-export types from chromadb.types
__all__ = ["Metadata", "Where", "WhereDocument", ]
META_KEY_CHROMA_DOCUMENT = "chroma:document"
T = TypeVar("T")
OneOrMany = Union[T, List[T]]

# URIs
URI = str
URIs = List[URI]
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


Document = str
Documents = List[Document]

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
    docs=  []
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

def create_document(df: DataFrame) -> List[str]:
    '''

    Args:
        df: Dataframe for a single code file

    Returns: Document for chromadb

    '''
    strings = []
    print('df.shape[0]', df.shape[0])
    print('df.shape[1]', df.shape[1])
    string_df = df.to_string(index=False)
    strings.append(string_df)
    print("string_df", string_df)
    print("type(string_df)", type(string_df))

    #print("texts:", texts)
    #print("type(texts):", type(texts))
    #print("--->document.page_content: ", texts.page_content)
    #docs.append(document)

    #print("document.page_content: ", texts.page_content)
    #try:
        #doc_embeddings = embedding_func(texts.page_content,embedding_model)
            #docs.append(document)
           # embeddings.append(doc_embeddings)
        #vector_db.add_document(document, doc_embeddings)
    #except Exception as e:
        #print(f"Failed to embed document: {e}")

    print("type(texts):", type(strings))

    return strings


def new_df(doc):

    string_df = doc[0] #.page_content
    print("string_df", string_df)
    print("len of string_df", len(string_df))

    rows = string_df[0].split('\n')
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

ImageDType = Union[np.uint, np.int64, np.float64]
Image = NDArray[ImageDType]
Images = List[Image]
Embeddable = Union[Documents, Images]
D = TypeVar("D", bound=Embeddable, contravariant=True)

def validate_embeddings(embeddings: Embeddings) -> Embeddings:
    """Validates embeddings to ensure it is a list of numpy arrays of ints, or floats"""
    if not isinstance(embeddings, (list, np.ndarray)):
        raise ValueError(
            f"Expected embeddings to be a list, got {type(embeddings).__name__}"
        )
    if len(embeddings) == 0:
        raise ValueError(
            f"Expected embeddings to be a list with at least one item, got {len(embeddings)} embeddings"
        )
    if not all([isinstance(e, np.ndarray) for e in embeddings]):
        raise ValueError(
            "Expected each embedding in the embeddings to be a numpy array, got "
            f"{list(set([type(e).__name__ for e in embeddings]))}"
        )
    for i, embedding in enumerate(embeddings):
        if embedding.ndim == 0:
            raise ValueError(
                f"Expected a 1-dimensional array, got a 0-dimensional array {embedding}"
            )
        if embedding.size == 0:
            raise ValueError(
                f"Expected each embedding in the embeddings to be a 1-dimensional numpy array with at least 1 int/float value. Got a 1-dimensional numpy array with no values at pos {i}"
            )
        if not all(
            [
                isinstance(value, (np.integer, float, np.floating))
                and not isinstance(value, bool)
                for value in embedding
            ]
        ):
            raise ValueError(
                "Expected each value in the embedding to be a int or float, got an embedding with "
                f"{list(set([type(value).__name__ for value in embedding]))} - {embedding}"
            )
    return embeddings

#PyEmbedding = PyVector
#PyEmbeddings = List[PyEmbedding]
Embedding = Vector

def normalize_embeddings(
    embeddings: Union[
        OneOrMany[Embedding],
        OneOrMany[Embedding],
    ]
) -> Embeddings:
    return cast(Embeddings, [np.array(embedding) for embedding in embeddings])

def maybe_cast_one_to_many_embedding(
    target: Union[OneOrMany[Embedding], OneOrMany[Embedding]]
) -> Embeddings:
    print("target", target)
    if isinstance(target, List):
        # One Embedding
        if isinstance(target[0], (int, float)):
            return cast(Embeddings, [target])
    elif isinstance(target, np.ndarray):
        if isinstance(target[0], (np.floating, np.integer)):
            return cast(Embeddings, [target])
    # Already a sequence
    return cast(Embeddings, target)

class EmbeddingFunction(Protocol[D]):
    def __call__(self, input: D) -> Embeddings:
        ...

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # Raise an exception if __call__ is not defined since it is expected to be defined
        call = getattr(cls, "__call__")

        def __call__(self: EmbeddingFunction[D], input: D) -> Embeddings:
            result = call(self, input)
            return validate_embeddings(
                normalize_embeddings(maybe_cast_one_to_many_embedding(result))
            )

        setattr(cls, "__call__", __call__)

    def embed_with_retries(
        self, input: D, **retry_kwargs: Dict[str, Any]
    ) -> Embeddings:
        return cast(Embeddings, retry(**retry_kwargs)(self.__call__)(input))


class ChromaGraphEmbedding(EmbeddingFunction):
    def __init__(self, codet_checkpoint, graph_checkpoint):
        self.model = GraphAutoEncoder(GAEncoder(PROPERTIES_DIM, EDGE_DIM, NUM_FEATURES, HIDDDEN_CHANNELS, OUT_CHANNELS), GADecoder(EDGE_DIM,IN_CHANNELS, HIDDDEN_CHANNELS, OUT_CHANNELS))#InnerProductDecoder())

        self.model_checkpoint = codet_checkpoint
        self.graph_checkpoint = graph_checkpoint

    def __call__(self, texts: Documents) -> chromadb.Embeddings:
        if not isinstance(texts, str):
            print(False)
        else:
            print(True)
        print("docs goes in ")
        print("type(texts): ", type(texts))
        #print("type(texts[0]): ", type(texts[0][0]))
        embeddings = []

        graph_model = self.model
        graph_model.load_state_dict(torch.load(self.graph_checkpoint))
        graph_model.eval()

        # regular calls send a string query sends the string in a list
        if isinstance(texts, list):
            if len(texts) == 1:
                texts = texts[0]
            else:
                print("error, large list input")

        df, df_nodes_relationship = new_df(docs)
        print("--->df", df)
        print("--->df_nodes_relationship", df_nodes_relationship)
        restructured_df = restructure_data(df, df_nodes_relationship, self.model_checkpoint)
        print("i , restructured_df:", restructured_df)

        graph_data = graph_positional_encoding([restructured_df])
        print("-->graph_data:", graph_data)


        z = graph_model.encode(graph_data[0].x.to(device), graph_data[0].x2.to(device), graph_data[0].edge_index.to(device),\
                               graph_data[0].edge_attr,graph_data[0].laplacian_eigenvector_pe.to(device))  # encodes the data

        print("z:", z)
        z = z.tolist()
        print("z to list:", z)

        #embeddings = List[np.ndarray] = [z]
        '''embeddings.append(z.tolist())
        print("embeddings:", embeddings)
        print("embeddings[0]:", embeddings[0])
        print("len(embeddings[0]):", len(embeddings[0]))
        print("len(embeddings[0][0]):", len(embeddings[0][0]))
        print("len(embeddings):", len(embeddings))
        z = np.array(z.tolist()).flatten().tolist()
        print("-->z:", z)
        print("-->type(z):", type(z))
        print("-->len(z):", len(z))
        print("==>z:", [z])
        print("==> type(z):", type(z))
        print("==>len(z):", len(z))
        #z = np.array(z.detach().numpy())
        #z = z.flatten()
        #embeddings: List[np.ndarray] = [z]
        #embeddings = [np.array(z.tolist())]
        #print("-->len(embeddings):", len(embeddings))
        print("type(embeddings)", type(embeddings))'''
        for embedding in z:
            if isinstance(embedding, np.ndarray):
                # Convert NumPy array to list of floats
                print("=>Here")
                embeddings.append(embedding.tolist())
            elif isinstance(embedding, list):
                # Ensure all elements are floats
                print("=>Hereeeee")
                print("embedding", embedding)
                print("type embedding", type(embedding))
                for x in embedding:
                    print("x:", x)
                #embeddings = [float(x) for x in embedding]
                embeddings.append([float(x) for x in embedding])
                print("=>embeddings", embeddings)
                print("=>type(embeddings)", type(embeddings))
                print("=>embeddings[0]", embeddings[0])
                print("=>type(embeddings[0])", type(embeddings[0]))
            else:
                print("=>else")
                raise ValueError(f"Unexpected embedding type: {type(embedding)}")
        print("embeddings", embeddings)
        return embeddings[0]


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
    docs = []
    for dataset in datasets:
        try:
            docs.append(create_document(dataset)
            )
        except Exception as e:
            print(f"Error adding dataset: {dataset}. Error: {str(e)}")

    print("--------> len(docs)", len(docs))
    #docs = Documents(docs)
    print("--------> docs", docs)
    #documents = [create_document(datasets[i]) for i in range(len(datasets))]
    #print("--------> len new_documents", len(new_documents))
    #documents = Documents(documents=[new_documents[0]])
    print("--------> len(documents)", len(docs))
    print("--------> type(documents)", type(docs))
    test_doc_0 = docs[0]
    print("--------> documents[0]", test_doc_0)
    print("--------> type(test_doc_0)", type(test_doc_0))
    print("--------> documents[0].page_content",  test_doc_0)
    #this = Documents(documents=test_doc_0)
    #print("--------> this", this)

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

    testing = embedding_function_chroma_graph(docs[0])
    #testing = embedding_function_chroma_graph(this)
    print("testing", testing)
    print("type(testing)", type(testing))
    #print("new shape ", np.array(testing).flatten().tolist())
    #print("another shape", np.array(testing).flatten().tolist().tolist())

    persistent_client = chromadb.PersistentClient()  # default settings

    print("persistent_client.list_collections():", persistent_client.list_collections())
    #del persistent_client
    #persistent_client = chromadb.PersistentClient()  # default settings
    # this gets the collection since it's already present
    collection = persistent_client.get_or_create_collection(name="graph_data_code",
                                                            embedding_function=embedding_function_chroma_graph)
    #print("collection", collection)
    print("There are", collection.count(), "in the collection")
    print("type(documents)", type(docs))
    for i, entry in enumerate(docs):
        print("i", i)
        print("entry", entry)
        embed = embedding_function_chroma_graph(entry)
        print("embed: ", embedding_function_chroma_graph(entry))
        embed0 = embed[0]
        embed_new = embed0.tolist()
        print("new embed: ", embed_new)
        collection.add(ids=f"{i}", embeddings=embed_new)
        #metadatas=entry.metadata, documents=entry.page_content)
        print(f"{i} of {len(docs)} added to db")

    results = collection.query(
        query_texts=[docs[0][0]],  # Chroma will embed this for you
        n_results=2  # how many results to return
    )
    print(results['distances'])
    print("End")

