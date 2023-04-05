import os
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from collections import Counter
from glob import glob

from doc_embed_torch import EMBED_VOCAB


def create_idf_vector():
    """Create an IDF vector for our corpus."""
    doc_counts = Counter()

    # Iterate over each document in the corpus
    print("Getting document counts...")
    file_paths = glob(os.path.join("data", "youtube", "raw", "*"))
    for filename in tqdm(file_paths):
        # Load the document
        with open(filename, "r") as f:
            doc = f.read()

        # Tokenize the document, and get the integer token ids
        token_ints = [EMBED_VOCAB.stoi[token] for token in word_tokenize(doc) if token in EMBED_VOCAB.stoi]

        # Update the document counts
        doc_counts.update(set(token_ints))

    print("Creating IDF vector...")
    idf_vector = []

    # Get the total number of documents
    num_docs = len(file_paths)

    # Create the IDF vector for the corpus
    for token_int in EMBED_VOCAB.stoi.values():
        doc_freq = doc_counts.get(token_int, 1)
        idf_vector.append(np.log(num_docs / doc_freq))

    print("Saving IDF vector...")
    # Save the IDF vector as a numpy array
    np.save(os.path.join("data", "idf_vector.npy"), np.array(idf_vector))


if __name__ == "__main__":
    create_idf_vector()
