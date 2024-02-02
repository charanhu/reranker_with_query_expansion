# reranker.py
from sentence_transformers import CrossEncoder
import numpy as np

def re_rank_documents_with_cross_encoder(query, retrieved_documents):
    # Initializing a Cross Encoder model for re-ranking
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Creating pairs of the original query and each retrieved document
    pairs = [[query, doc] for doc in retrieved_documents]
    # Predicting scores using the Cross Encoder
    scores = cross_encoder.predict(pairs)

    # Printing the scores of each document
    print("Scores:")
    for score in scores:
        print(score)

    # Printing the new ordering of documents based on the scores
    print("New Ordering:")
    for o in np.argsort(scores)[::-1]:
        print(o + 1)
