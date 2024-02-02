# main.py
# Suppressing warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Importing necessary modules and functions from external files
from helper_utils import load_chroma, word_wrap, project_embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import CrossEncoder
import numpy as np

# Importing functions from created files
from reranker import re_rank_documents_with_cross_encoder
from query_expansion import generate_question_response

# Function to load data and count the number of records
def load_and_count_data(filename, collection_name, embedding_function):
    # Loading collection from a file using helper function
    data_collection = load_chroma(filename=filename, collection_name=collection_name, embedding_function=embedding_function)
    # Counting the number of records in the collection
    count = data_collection.count()
    # Returning the collection and the count
    return data_collection, count

# Function to query data collection and retrieve documents based on a query
def query_and_retrieve_documents(data_collection, query, n_results=10):
    # Querying collection for documents related to the given query
    results = data_collection.query(query_texts=query, n_results=n_results, include=['documents', 'embeddings'])
    # Retrieving the documents from the results
    retrieved_documents = results['documents'][0]
    # Returning the retrieved documents
    return retrieved_documents

# Function to print documents and their embeddings
def print_documents_and_embeddings(retrieved_documents):
    # Printing each document after applying word wrapping for better display
    for document in retrieved_documents:
        print(word_wrap(document))
        print('')

# Example usage:
# Initializing the SentenceTransformer embedding function
embedding_function = SentenceTransformerEmbeddingFunction()

# Loading data and counting the number of records
data_collection, count = load_and_count_data(filename='test.pdf',
                                            collection_name='test_3',
                                            embedding_function=embedding_function)

print("=====================================")
print(f"Data Collection Count: {count}\n")

# Re-ranking the long tail
query = "Explain the methodology and core architecture used in this project"
retrieved_documents = query_and_retrieve_documents(data_collection, query)
print("=====================================")
print("Retrieved Documents:")
print_documents_and_embeddings(retrieved_documents)
re_rank_documents_with_cross_encoder(query, retrieved_documents)

# Automatic Query Expansion and Re-ranking
input_text = "Explain the methodology and core architecture used in this project"
response_list = generate_question_response(input_text)
print("=====================================")
print("Response List:")
print(response_list)

# Reranking the documents with the new queries
for query in response_list:
    retrieved_documents = query_and_retrieve_documents(data_collection, query)
    print("=====================================")
    print(f"Retrieved Documents for Query: {query}")
    print_documents_and_embeddings(retrieved_documents)
    re_rank_documents_with_cross_encoder(query, retrieved_documents)
