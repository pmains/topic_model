import argparse
import os
from collections import defaultdict
import time
from random import sample, seed

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tqdm import tqdm
from umap import UMAP

from torch.utils.data import Dataset


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
        self.file_paths = [os.path.join(topic_dir, f) for f in os.listdir(topic_dir) if f.endswith(".txt")]

        # Map video IDs to file paths
        self.video_id_to_file_path = defaultdict(list)
        for file_path in self.file_paths:
            file_name = os.path.basename(file_path)
            # Video ID is everything before the last underscore
            video_id = file_name[:file_name.rfind("_")]
            self.video_id_to_file_path[video_id].append(file_path)

    def get_file_paths(self, video_id):
        """
        Return the text file paths for a given video ID
        :param video_id: str in format {channel_id}_{video_id}
        :return: list of file paths
        """
        return self.video_id_to_file_path[video_id]

    def __len__(self):
        """Returns the number of chunks in the dataset"""
        return len(self.file_paths)

    def __getitem__(self, *indices):
        """
        Accepts a list of indices and returns a list of text chunks.
        If only one index is passed, returns a single text chunk.
        """

        items = []

        for index in indices:
            # Read the text from the file
            with open(self.file_paths[index], "r") as f:
                chunk_text = f.read()
            items.append(chunk_text)

        if len(items) == 1:
            return items[0]
        return items


class TopicModeler:
    def __init__(self, era_df: pd.DataFrame, chunk_size: int, max_files: int = 1000):
        """
        :param era_df: DataFrame containing era information
        :param chunk_size: Number of words to include in each chunk
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
            Whether or not to make a visualization of the topic model

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

        # Create a dataframe to hold the text chunks
        text_df = pd.DataFrame(columns=['video_id', 'chunk', 'text'])
        # Remove leading @ symbols from channel IDs
        topic_df['channel_id'] = topic_df['channel_id'].str.replace('@', '')
        # Combine the video ID and channel ID to retrieve chunks for each video from the dataset
        # topic_df['channel_video_id'] = topic_df['channel_id'] + '_' + topic_df['video_id']

        # Get the text chunks for each video
        print("Getting text chunks for each video...")
        for video_id in tqdm(topic_df['video_id'].values):
            # Get the file paths for the video's text chunks
            chunk_paths = self.chunk_dataset.get_file_paths(video_id)
            if len(chunk_paths) == 0:
                continue
            # Read the text from each chunk and add it to the dataframe
            # print(f"Reading {len(chunk_paths)} chunks for {video_id}...")
            video_text_data = list()
            for chunk_path in chunk_paths:
                with open(chunk_path, 'r') as f:
                    chunk_text = f.read()
                video_text_data.append({'video_id': video_id, 'chunk': chunk_path, 'text': chunk_text})

            # Append the video's text chunks to our dataframe
            text_df = text_df.append(video_text_data, ignore_index=True)

        # Reset df to only include documents with text
        text_df = text_df[text_df['text'] != '']
        text_df = text_df.dropna(subset=['text'])

        # Merge the text_df with topics_df
        topic_df = topic_df.merge(text_df, on='video_id', how='left')

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

        """
        if len(topic_df) < 600:
            mts = int(round(len(topic_df) * 0.1, 0))
        else:
            mts = 75
        """

        # Initialize the BERTopic model and fit it to the text
        print("Training the BERTopic model...")
        # Get time to run fit_transform
        start_time = time.time()
        topic_model = BERTopic(
            vectorizer_model=vectorizer_model, top_n_words=5, nr_topics=9, umap_model=umap_model
        )

        # If there are more than self.max_files documents, sample self.max_files documents
        if len(topic_df) > self.max_files:
            # Get self.max_files sample indices from the dataframe
            seed(42)
            sample_indices = sample(topic_df.index.tolist(), self.max_files)
            # Get the documents from the sample indices
            topic_df = topic_df.loc[sample_indices, :].copy()
            documents = topic_df.loc[sample_indices, 'text'].astype(str).tolist()
        else:
            documents = topic_df['text'].astype(str).tolist()

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
        np.save(os.path.join("data", "topics", f"{era}_{category}_topic_embeddings.npy"), topic_embeddings)
        # Save topic info
        topic_model_df.to_csv(os.path.join("data", "topics", f"{era}_{category}_topic_info.csv"), index=False)

        # Make sure data/viz exists
        os.makedirs(os.path.join("data", "viz"), exist_ok=True)

        if make_viz:
            doc_fig = topic_model.visualize_documents(
                topic_df['text'].astype(str).tolist(),
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
    parser.add_argument("--max_files", type=int, default=1000)
    parsed_args = parser.parse_args()

    # Set era_df using os.path.join
    video_df = pd.read_csv(os.path.join("data", "combined_data.csv"))
    my_chunk_size = 128
    house_eras = (
        'JB-H2', 'JB-H1', 'DT-H4', 'JB-H3', 'DT-H3', 'DT-H2', 'DT-H1', 'BO-H8', 'BO-H7', 'BO-H6', 'BO-H5', 'BO-H4',
        'BO-H3', 'BO-H2', 'BO-H1', 'GB-H3', 'GB-H2', 'GB-H1'
    )

    categories = (DEMOCRAT, REPUBLICAN)

    topic_modeler = TopicModeler(video_df, chunk_size=my_chunk_size, max_files=parsed_args.max_files)
    for era in house_eras:
        for category in categories:
            print(f"Creating topic model for {era} {category}...")
            topic_modeler.create_topic_model(era=era, era_type=HOUSE, category=category, make_viz=True)
