import argparse
import os
from collections import defaultdict
import time
from random import sample, seed

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from umap import UMAP

from torch.utils.data import Dataset

from doc_embed_torch import DocumentEmbeddingTrainer, load_run_config, add_to_batch

PRESIDENTIAL = 'presidential'
HOUSE = 'house'

REPUBLICAN = 'republican'
DEMOCRAT = 'democrat'
INDEPENDENT = 'independent'
ALL = 'all'


class ChunkDataset(Dataset):
    """Dataset for chunks of text"""

    def __init__(self, chunk_size):
        self.chunk_size = chunk_size

        # Get all .txt file paths in data/train-{chunk_size} and data/test-{chunk_size} directories
        # Scan data/train-{chunk_size} directory
        topic_dir = os.path.join("data", f"topic-{chunk_size}")
        # Combine the train and test files
        self.text_files = [os.path.join(topic_dir, f) for f in os.listdir(topic_dir) if f.endswith(".txt")]
        self.token_files = [f.replace(".txt", ".pt") for f in self.text_files]

        # Map video IDs to text file paths
        self.video_id_to_text_files = defaultdict(list)
        for file_path in self.text_files:
            file_name = os.path.basename(file_path)
            # Video ID is everything before the last underscore
            video_id = file_name[:file_name.rfind("_")]
            self.video_id_to_text_files[video_id].append(file_path)

        # Map video IDs to token file paths
        self.video_id_to_token_files = defaultdict(list)
        for file_path in self.token_files:
            file_name = os.path.basename(file_path)
            # Video ID is everything before the last underscore
            video_id = file_name[:file_name.rfind("_")]
            self.video_id_to_token_files[video_id].append(file_path)

    def get_text_files(self, video_id):
        """
        Return the text file paths for a given video ID
        :param video_id: str in format {channel_id}_{video_id}
        :return: list of file paths
        """
        return self.video_id_to_text_files[video_id]

    def get_token_files(self, video_id):
        """
        Return the token file paths for a given video ID
        :param video_id: str in format {channel_id}_{video_id}
        :return: list of file paths
        """
        return self.video_id_to_token_files[video_id]

    def document_file_names(self, sample_size=None):
        """Get document test and token file paths"""
        if sample_size is None or sample_size >= len(self.text_files):
            return self.text_files, self.token_files

        # Get a random sample of documents
        seed(1337)
        # Get a random sample of documents
        sample_indices = sample(range(len(self.text_files)), sample_size)
        sample_text_files = [self.text_files[i] for i in sample_indices]
        sample_token_files = [self.token_files[i] for i in sample_indices]
        return sample_text_files, sample_token_files

    def __len__(self):
        """Returns the number of chunks in the dataset"""
        return len(self.text_files)

    def __getitem__(self, *indices):
        """
        Accepts a list of indices and returns a list of text chunks.
        If only one index is passed, returns a single text chunk.
        """

        items = []

        for index in indices:
            # Read the text from the file
            with open(self.text_files[index], "r") as f:
                chunk_text = f.read()
            items.append(chunk_text)

        if len(items) == 1:
            return items[0]
        return items


class TopicModeler:
    def __init__(self, era_df, chunk_size, max_files=1000, embedder=None, run_code=None):
        """
        :param era_df: DataFrame containing era information
        :param chunk_size: Number of words to include in each chunk
        :param embedder: Embedding model to use
        """

        # Create a list of stop words
        custom_stop_words = [
            'dont', "don't", 'going', 'im', "i'm", 'just', 'know', 'like', 'okay', 'really', 'right', 'so', 'thank',
            'that', "that's", "thats", 'thing', 'think', 'uh', 'um', 'want', 'well', 'yeah', 'you', 'nan'
        ]
        extended_stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)
        self.stop_words = list(extended_stop_words)

        self.video_df = era_df
        self.chunk_size = chunk_size
        self.doc_path = os.path.join("data", f"train-{chunk_size}")
        self.max_files = max_files
        self.embedder = embedder
        self.run_code = run_code

        # Create our dataset, which will return the text chunks for each video
        print("Creating dataset...")
        self.chunk_dataset = ChunkDataset(self.chunk_size)

    def create_topic_model(self, era: str, era_type: str, category: str, make_viz: bool = False) -> pd.DataFrame:
        """
        Takes a dataframe and era, era_type, and category and returns a topic model dataframe

        Parameters
        ----------
        era : str
            The era (president name or congress code) to filter the dataframe by
        era_type : str
            The era_type (presidential or congressional) to filter the dataframe by
        category : str
            The political party to filter the dataframe by
        make_viz : bool
            Whether to make a visualization of the topic model

        Returns
        -------
        pd.DataFrame
            Adds 'topic' and 'prob' to the given dataframe
        """

        topic_df = self.video_df.copy()

        # Filter the dataframe by era
        print("Filtering videos...")
        if era_type == PRESIDENTIAL:
            topic_df = topic_df[topic_df['presidential_era'] == era].copy()
        elif era_type == HOUSE:
            topic_df = topic_df[topic_df['house_era'] == era].copy()
        else:
            print('Invalid era_type')

        # Filter the dataframe by party if category is not 'all'
        if category == REPUBLICAN:
            topic_df = topic_df[topic_df['party'] == 'REPUBLICAN'].copy()
        elif category == DEMOCRAT:
            topic_df = topic_df[topic_df['party'] == 'DEMOCRATIC'].copy()
        elif category == INDEPENDENT:
            topic_df = topic_df[topic_df['party'] == 'INDEPENDENT'].copy()
        elif category == ALL:
            pass
        else:
            print('Invalid category')
            # exit the function
            return topic_df

        # Print the number of documents remaining in the dataframe
        print(f'{len(topic_df)} documents in {era} {era_type} {category} dataframe')
        # If there are no documents, just return the dataframe as is
        if len(topic_df) == 0:
            return topic_df

        # Remove leading @ symbols from channel IDs
        topic_df['channel_id'] = topic_df['channel_id'].str.replace('@', '')
        # Combine the video ID and channel ID to retrieve chunks for each video from the dataset
        # topic_df['channel_video_id'] = topic_df['channel_id'] + '_' + topic_df['video_id']

        # Get the text chunks for each video
        print("Getting text chunks for each video...")
        chunk_text_paths = list()
        video_text_data = list()
        for video_id in tqdm(topic_df['video_id'].values):
            # Get the file paths for the video's text chunks
            video_text_paths = self.chunk_dataset.get_text_files(video_id)
            # Add the video's text chunks to our list of text chunks
            chunk_text_paths.extend(video_text_paths)
            # Add the video's text chunks to our dataframe
            for chunk_text_path in video_text_paths:
                video_text_data.append({'video_id': video_id, 'chunk': chunk_text_path})

        text_df = pd.DataFrame(video_text_data)

        # If we have more than max_files, randomly sample the text chunks
        if self.max_files is not None and len(chunk_text_paths) > self.max_files:
            # Randomly sample the text chunks
            chunk_text_paths = sample(chunk_text_paths, self.max_files)
            # Filter the video text dataframe to only include the sampled chunks
            text_df = text_df[text_df['chunk'].isin(chunk_text_paths)].copy()

        # Create a list of text chunks that we will add as a column to our dataframe
        print("Reading text files ...")
        text_chunks = list()
        for idx, chunk_text_path in enumerate(chunk_text_paths):
            with open(chunk_text_path, 'r') as f:
                text_chunks.append(f.read())
        text_df['text'] = text_chunks

        # If we have an embedder, get the embeddings for each chunk and make a column of our dataframe
        if self.embedder is not None:
            print("Embedding documents ..")
            # Iterate over the text chunks in batches of 64
            chunk_embeddings = None
            batch_size = 16
            for i in tqdm(range(0, len(chunk_text_paths), batch_size)):
                # Get the embeddings for the current batch
                batch_text_paths = chunk_text_paths[i:i + batch_size]
                # Get the corresponding paths for the token files as a tensor
                batch_token_paths = [path.replace('.txt', '.pt') for path in batch_text_paths]
                # Read the embeddings from the files and place them in a tensor
                batch_tokens = [torch.load(path) for path in batch_token_paths]
                batch_token_tensor = torch.stack(batch_tokens)

                # Get the embeddings for the current batch
                batch_embeddings = self.embedder(batch_token_tensor, return_doc_embedding=True)
                del batch_tokens, batch_token_tensor
                if chunk_embeddings is None:
                    chunk_embeddings = batch_embeddings
                else:
                    chunk_embeddings = torch.cat((chunk_embeddings, batch_embeddings))
                del batch_embeddings

            # Add the embeddings to the dataframe
            print("Saving embeddings")
            text_df['embedding'] = chunk_embeddings.detach().numpy().tolist()
            del chunk_embeddings

        # Reset df to only include documents with text
        text_df = text_df[text_df['text'] != '']
        text_df = text_df.dropna(subset=['text'])

        # Merge the text_df with topics_df
        print("Merging our data ...")
        topic_df = topic_df.merge(text_df, on='video_id', how='right')
        del text_df

        # # Create a topic model

        # Initialize the CountVectorizer with the extended stop words
        # vectorizer_model = CountVectorizer(stop_words=extended_stop_words)

        # Create an instance of TfidfVectorizer with custom parameters
        print("Training the TfidfVectorizer...")
        vectorizer_model = TfidfVectorizer(
            stop_words=self.stop_words, ngram_range=(1, 2), max_df=0.75, min_df=.05
        )
        vectorizer_model.fit(topic_df['text'].astype(str).tolist())

        # I'm not sure why UMAP can't be imported directly, but this works
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

        # Initialize the BERTopic model and fit it to the text
        print("Training the BERTopic model...")
        # Get time to run fit_transform
        start_time = time.time()
        topic_model = BERTopic(
            vectorizer_model=vectorizer_model, top_n_words=5, nr_topics=9, umap_model=umap_model
        )

        documents = topic_df['text'].astype(str).tolist()
        if self.embedder is not None:
            # Get the topic embeddings as a 2D numpy array
            embeddings = np.array(topic_df.embedding.values.tolist())
            topics, probs = topic_model.fit_transform(documents=documents, embeddings=embeddings)
        else:
            embeddings = None
            topics, probs = topic_model.fit_transform(documents=documents)
        print(f"Time to run fit_transform: {time.time() - start_time} seconds")

        # Get the topic info
        topic_model_df = topic_model.get_topic_info()
        # Add the topic to the original dataframe
        topic_df['topic'] = topics
        topic_df['prob'] = probs
        topic_df = topic_df.merge(topic_model_df, left_on='topic', right_on='Topic', how='left')

        # Make sure data/topics exists
        os.makedirs(os.path.join("data", "topics"), exist_ok=True)
        # Save topic embeddings
        topic_embeddings = np.array(topic_model.topic_embeddings_)
        if self.run_code is not None:
            # Save topic embeddings
            np.save(os.path.join("data", "topics", f"{era}_{category}_topic_embeddings_{self.run_code}.npy"),
                    topic_embeddings)

            # Save topic info
            topic_model_df.to_csv(os.path.join("data", "topics", f"{era}_{category}_topic_info_{self.run_code}.csv"),
                                  index=False)
        else:
            np.save(os.path.join("data", "topics", f"{era}_{category}_topic_embeddings.npy"), topic_embeddings)
            # Save topic info
            topic_model_df.to_csv(os.path.join("data", "topics", f"{era}_{category}_topic_info.csv"), index=False)

        # Make sure data/viz exists
        os.makedirs(os.path.join("data", "viz"), exist_ok=True)

        if make_viz:
            doc_fig = topic_model.visualize_documents(
                topic_df['text'].astype(str).tolist(),
                embeddings=embeddings,
                title=f'<b>{era} Era {category} Topics</b>',
                height=600,
                width=1000
            )
            doc_fig.write_html(os.path.join("data", "viz", f"{era}_{category}_topics.html"))

            topics_over_time = topic_model.topics_over_time(
                topic_df['text'].astype(str).tolist(), topic_df['date'].astype(str).tolist()
            )
            topic_fig = topic_model.visualize_topics_over_time(topics_over_time)
            topic_fig.write_html(os.path.join("data", "viz", f"{era}_{category}_topics_over_time.html"))

        return topic_df


if __name__ == "__main__":
    # Get max_files from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=1000)
    parser.add_argument("--model-type", type=str, default="dual")
    # Load model to generate embeddings
    parser.add_argument("--run-code", type=str, default=None)
    parsed_args = parser.parse_args()

    # Set era_df using os.path.join
    video_df = pd.read_csv(os.path.join("data", "combined_data.csv"))
    house_eras = (
        'JB-H2', 'JB-H1', 'DT-H4', 'JB-H3', 'DT-H3', 'DT-H2', 'DT-H1', 'BO-H8', 'BO-H7', 'BO-H6', 'BO-H5', 'BO-H4',
        'BO-H3', 'BO-H2', 'BO-H1'
    )

    categories = (DEMOCRAT, REPUBLICAN)

    if parsed_args.run_code is not None:
        run_config = load_run_config(parsed_args.run_code)
        embed_trainer = DocumentEmbeddingTrainer(run_code=parsed_args.run_code, model_type=parsed_args.model_type)
        embed_trainer.load_model(run_code=parsed_args.run_code)
        my_chunk_size = run_config['chunk_size'].item()
        my_model = embed_trainer.model
    else:
        my_model = None
        my_chunk_size = parsed_args.chunk_size

    topic_modeler = TopicModeler(
        video_df, chunk_size=my_chunk_size, max_files=parsed_args.max_files, embedder=my_model,
        run_code=parsed_args.run_code
    )
    for my_era in house_eras:
        for my_category in categories:
            print(f"Creating topic model for {my_era} {my_category}...")
            topic_modeler.create_topic_model(era=my_era, era_type=HOUSE, category=my_category, make_viz=True)
