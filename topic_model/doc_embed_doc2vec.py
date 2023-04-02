import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from reiterable import Reiterable

MODEL_NAME = os.path.join('models', 'youtube_model')
CHUNK_SIZE = 4096


def read_and_preprocess(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    words = word_tokenize(content.lower())  # tokenize and lowercase words
                    for i in range(0, len(words), CHUNK_SIZE):
                        chunk_words = words[i:i+CHUNK_SIZE]
                        yield TaggedDocument(words=chunk_words, tags=[file, i])


def train():
    doc_folder_path = 'data/youtube/raw'
    # tagged_docs = list(read_and_preprocess(doc_folder_path))

    # vector_size = 100
    vector_size = 4096
    window = 5
    min_count = 5
    epochs = 20

    tagged_docs = Reiterable(read_and_preprocess, doc_folder_path)

    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=os.cpu_count())
    print("Building vocabulary...")
    model.build_vocab(tagged_docs)

    print("Training...")
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=epochs)

    model.save(MODEL_NAME)


def load():
    return Doc2Vec.load(MODEL_NAME)


def similarity(doc_name):
    # Retrieve the document vector for a specific document
    model = load()
    doc_vector = model.dv[doc_name]

    return model.dv.most_similar([doc_vector], topn=10)


def vector(doc_name):
    # Retrieve the document vector for a specific document
    model = load()
    return model.dv[doc_name]


if __name__ == '__main__':
    """
    Usage:
    python document_embedder.py --train
    python document_embedder.py --similarity <doc_name>
    python document_embedder.py --vector <doc_name>
    """

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--similarity', action='store', default=None)
    parser.add_argument('--vector', action='store', default=None)
    args = parser.parse_args()

    if args.train:
        train()
    elif args.similarity:
        print(similarity(args.similarity))
    elif args.vector:
        print(vector(args.vector))
