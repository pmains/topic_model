"""
Usage:
    Break our documents into chunks, and save them as PyTorch tensors:
    python doc_embed_torch.py --chunk --chunk_size [chunk_size]

    Train the model:
    python doc_embed_torch.py --train [--args]

    Reinforce the model:
    python doc_embed_torch.py --reinforce [--args]

    --help for more info
"""

import argparse
import os
import string
from random import random, shuffle, choice, sample

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from torchtext import vocab
from tqdm import tqdm


# Create unique 8-digit alphanumeric code for this run
def gen_run_code():
    alphabet = string.ascii_letters + string.digits
    return ''.join(choice(alphabet) for _ in range(8))


# Load the GloVe embedding vocabulary
EMBED_VOCAB = vocab.GloVe(name='6B', dim=100)


# Add the [MASK] token to the vocabulary if it doesn't exist
def add_token_to_vocab(token, embedding_size=100):
    if token not in EMBED_VOCAB.stoi:
        EMBED_VOCAB.itos.append(token)
        EMBED_VOCAB.stoi[token] = len(EMBED_VOCAB.itos)-1

        # Make sure data/vocab exists
        if not os.path.exists(os.path.join("data", "vocab")):
            os.mkdir(os.path.join("data", "vocab"))

        # Check if the vector for the token already exists
        vector_path = os.path.join("data", "vocab", f"{token}.pt")
        if os.path.exists(vector_path):
            # Load the vector from a file
            with open(vector_path, "rb") as f:
                new_vector = torch.load(f)
        else:
            new_vector = torch.randn(embedding_size)
            # Save the new vector to a file
            with open(os.path.join("data", "vocab", f"{token}.pt"), "wb") as f:
                torch.save(new_vector, f)

        EMBED_VOCAB.vectors = torch.cat([EMBED_VOCAB.vectors, new_vector.unsqueeze(0)])
        return EMBED_VOCAB.stoi[token]

    return EMBED_VOCAB.stoi[token]


# Add the [UNK], [PAD], [MASK] and [CLS] token ids to the vocabulary
UNK_TOKEN_ID = add_token_to_vocab('[UNK]')
PAD_TOKEN_ID = add_token_to_vocab('[PAD]')
MASK_TOKEN_ID = add_token_to_vocab('[MASK]')
CLS_TOKEN_ID = add_token_to_vocab('[CLS]')

VOCAB_SIZE = len(EMBED_VOCAB.itos)


class DocumentDataset(Dataset):
    """
    Class for loading a document dataset. Documents are assumed
    to be in a folder, with each document in a separate file.
    """

    def __init__(self, folder_path):
        """Set folder path and vocab"""
        self.folder_path = folder_path
        # Get a list of all files in the folder
        self.file_list = os.listdir(self.folder_path)
        # Put the files in random order
        shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """Returns a list of word indices for a document"""
        file_path = os.path.join(self.folder_path, self.file_list[idx])
        token_ids = torch.load(file_path)

        return token_ids


class PredictChunkDataset(Dataset):
    def __init__(self, dir_path, mask_prob=0.15):
        """
        Files in our directory will have a name in format "{channel}-{video_id}_{chunk_id}.pt"
        The chunk_id values are sequential, so we can use them to sort the files
        Files are only considered sequential if they belong to the same channel and video_id
        So we first group the files by channel and video_id, and then sort them by chunk_id
        Only a chunk with a subsequent chunk_id can be used to predict the next chunk

        :param dir_path: Path to the directory containing the dataset
        """

        self.dir_path = dir_path  # Path to the directory containing the dataset
        self.chunk_list = []
        self.mask_prob = mask_prob

        def get_video(file_name):
            last_underscore = file_name.rfind('_')
            file_video_id = file_name[:last_underscore]
            chunk_id = int(file_name[last_underscore + 1:-3])

            return file_video_id, chunk_id

        # Get a list of all files in the directory and create a dataframe
        print("Grouping files by channel and video_id ...")
        file_list = os.listdir(self.dir_path)
        file_df = pd.DataFrame(file_list, columns=['file_name'])
        file_df[['video_id', 'chunk_id']] = file_df.file_name.apply(lambda x: pd.Series(get_video(x)))
        # Group the files by channel and video_id
        grouped = file_df.groupby(['video_id'])

        # Iterate and add each subsequent pair of chunks to the chunk_list
        print("Iterating over videos to find subsequent chunks ...")
        for video_id, group in tqdm(grouped):
            group = group.sort_values('chunk_id')
            for i in range(len(group) - 1):
                # Add the input and predict chunk file names to the chunk_list
                input_chunk_file = group.iloc[i].file_name
                # Randomly choose the next chunk or a random chunk to predict
                if random() < 0.5:
                    # Use the next chunk as the predict chunk
                    predict_chunk_file = group.iloc[i + 1].file_name
                    is_next_chunk = True
                else:
                    # Use a random chunk as the predict chunk
                    predict_chunk_file = sample(list(group.file_name), 1)[0]
                    is_next_chunk = False
                self.chunk_list.append((input_chunk_file, predict_chunk_file, is_next_chunk))

    def __len__(self):
        return len(self.chunk_list)  # Return the total number of samples in the dataset

    def mask_tokens(self, token_ids):
        masked_token_ids = token_ids.clone()  # Create a copy of token ids

        # Iterate through the token ids and mask tokens with a probability of self.mask_prob
        for i, token_id in enumerate(token_ids):
            if random() < self.mask_prob:
                masked_token_ids[i] = MASK_TOKEN_ID  # Replace the token id with mask_token_id

        return masked_token_ids

    def __getitem__(self, idx):
        # Get the input and predict chunk file names, and a flag indicating if the chunk is the next one
        input_chunk_file, predict_chunk_file, is_next_chunk = self.chunk_list[idx]
        # Load the token ids from the file
        input_chunk = torch.load(os.path.join(self.dir_path, input_chunk_file))
        # Mask some of the tokens in the chunk
        masked_chunk = self.mask_tokens(input_chunk)
        # Load the token ids for the chunk we want to predict
        predict_chunk = torch.load(os.path.join(self.dir_path, predict_chunk_file))

        return input_chunk, masked_chunk, predict_chunk, is_next_chunk


class DocumentEmbedder(nn.Module):
    """Embeds documents using a masked language model for training"""

    def __init__(self, vocab_size, embedding_size, num_heads, dim_feedforward, num_layers):
        super(DocumentEmbedder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            activation='gelu',
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
            batch_first=True,
        )
        self.fc = nn.Linear(embedding_size, vocab_size)

        # Layer to predict whether the next chunk is a continuation of the current chunk
        self.next_chunk_classifier = nn.Sequential(
            nn.Linear(embedding_size*4, 2),
            nn.Softmax(dim=-1),
        )

        self.idf_weights = None

    def forward(self, chunk, masked_chunk=None, next_chunk=None, return_doc_embedding=False):
        """
        Take document, masked document, and next chunk as input
        """

        embedded_chunk = self.embedding(chunk)
        encoded_chunk = self.transformer(embedded_chunk, embedded_chunk)
        doc_embedding = self.get_doc_embedding(chunk, encoded_chunk)

        if return_doc_embedding:
            return doc_embedding

        # Predict the masked tokens
        embedded_masked_chunk = self.embedding(masked_chunk)
        encoded_masked_chunk = self.transformer(embedded_masked_chunk, embedded_masked_chunk)
        masked_logits = self.fc(encoded_masked_chunk)

        embedded_next_chunk = self.embedding(next_chunk)
        encoded_next_chunk = self.transformer(embedded_next_chunk, embedded_next_chunk)
        next_chunk_embedding = self.get_doc_embedding(next_chunk, encoded_next_chunk)
        # Predict the next chunk
        next_chunk_prob = self.next_chunk_prediction(doc_embedding, next_chunk_embedding)

        return masked_logits, next_chunk_prob

    def next_chunk_prediction(self, doc_embedding, next_chunk_embedding):
        concatenated_embeddings = torch.cat((doc_embedding, next_chunk_embedding), dim=1)
        next_chunk_prob = self.next_chunk_classifier(concatenated_embeddings)
        return next_chunk_prob

    def get_doc_embedding(self, x, encoding):
        # Load the IDF weights if they haven't been loaded yet
        if self.idf_weights is None:
            self.idf_weights = np.load(os.path.join("data", "idf_vector.npy"))

        doc_embeddings = None
        # Iterate through the vectors of x and encoding, each of which represents a document
        for (x_vec, encoding_vec) in zip(x, encoding):
            # Get the IDF weights of the tokens in the document
            idf_weights = torch.tensor(self.idf_weights[x_vec])
            # Normalize IDF weights and repeat them to match the shape of the encoding vectors
            idf_weights = (idf_weights / idf_weights.sum()).unsqueeze(1).repeat(1, encoding_vec.shape[1])
            # Get the average of the encoding vectors weighted by the IDF weights
            mean_embedding = (encoding_vec * idf_weights).mean(dim=0)
            max_embedding = encoding_vec.max(dim=0)[0]
            # min_embedding = encoding_vec.min(dim=0)[0]
            # std_embedding = encoding_vec.std(dim=0)
            # Concatenate the mean, max, min, and std embeddings
            doc_embedding_vec = torch.cat((mean_embedding, max_embedding)).unsqueeze(0)
            # doc_embedding_vec = torch.cat((mean_embedding, max_embedding, min_embedding, std_embedding)).unsqueeze(0)

            # Add the document embedding to the doc_embeddings tensor
            if doc_embeddings is None:
                doc_embeddings = doc_embedding_vec
            else:
                doc_embeddings = torch.cat((doc_embeddings, doc_embedding_vec))

        return doc_embeddings.to(encoding.dtype)


class Int8MinMaxObserver(torch.quantization.MinMaxObserver):
    """Custom observer for INT8 quantization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_min = 0
        self.quant_max = 255
        self.dtype = torch.quint8
        self.qscheme = torch.per_tensor_affine
        self.reduce_range = False


class DocumentEmbeddingTrainer:
    """This class handles training and evaluation of the document embedding model"""

    def __init__(self, chunk_size=None, embedding_size=None, run_code=None):
        if run_code is None:
            self.run_code = gen_run_code()
            self.chunk_size = chunk_size
            self.embedding_size = embedding_size
        else:
            self.run_code = run_code
            run_config = load_run_config(run_code)
            self.chunk_size = run_config.chunk_size
            self.embedding_size = run_config.embedding_size

        self.model = None
        self.adversary = None
        self.eps = None

        # Hyper-parameters
        self.mlm_lr = None
        self.mlm_epochs = None
        self.mlm_batch_size = None

        self.adversary_lr = None
        self.adversary_epochs = None
        self.adversary_batch_size = None

        self.combined_epochs = None

        # Train the DocumentEmbeddingTransformer to generate document embeddings.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        chunk_dir = os.path.join("data", f"train-{self.chunk_size}")
        self.doc_dataset = DocumentDataset(chunk_dir)
        self.predict_dataset = PredictChunkDataset(chunk_dir)

    def init_model(self, batch_size, num_epochs, num_heads, dim_feedforward, num_layers, lr):
        # Create the DocumentMLMEmbedder model
        self.mlm_lr = lr
        self.mlm_epochs = num_epochs
        self.mlm_batch_size = batch_size

        print("Creating the MLM model ...")
        self.model = DocumentEmbedder(
            vocab_size=VOCAB_SIZE,
            embedding_size=self.embedding_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
        ).to(self.device)

    @staticmethod
    def calc_loss(logits, targets, loss_function):
        # Calculate the loss
        batch_size, seq_len = targets.shape
        logits_view = logits.view(batch_size * seq_len, -1)
        labels_view = targets.view(batch_size * seq_len)
        loss = loss_function(logits_view, labels_view)
        return loss

    @staticmethod
    def add_to_batch(batch, vector):
        if batch is None:
            batch = vector.unsqueeze(0)
        else:
            batch = torch.cat([batch, vector.unsqueeze(0)])
        return batch

    def train(self, batch_size=None, epochs=None, lr=None):
        """Train the DocumentMLMEmbedder model"""

        if self.model is None:
            raise ValueError("The model has not been initialized. Call init_mlm() or load_mlm() first.")

        # Set the batch size, epochs, and learning rate
        if batch_size is not None:
            self.mlm_batch_size = batch_size
        if epochs is not None:
            self.mlm_epochs = epochs
        if lr is not None:
            self.mlm_lr = lr

        # Prepare the model for quantization
        self.prepare_for_quantization()
        self.model.to(self.device)
        self.model.train()

        # Create the optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.mlm_lr)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Read the loss from a file if it exists
        new_loss_df = pd.DataFrame(columns=["run_code", "epoch", "loss"])

        # Sample [self.mlm_epochs * self.mlm_batch_size] random documents from the training dataset
        sample_indices = sample(range(len(self.predict_dataset)), self.mlm_epochs * self.mlm_batch_size)

        # Train the model
        print("Training the MLM model ...")
        for epoch in tqdm(range(self.mlm_epochs)):
            total_loss = 0
            num_batches = 0

            optimizer.zero_grad()
            # The input tokens for the first chunk of each pair
            batch_chunks = None
            # The masked tokens for the first chunk of each pair
            batch_masked_chunks = None
            # The target tokens for the second chunk of each pair
            batch_next_chunks = None
            # Whether the second chunk is the next chunk in the document
            batch_is_next_chunk = []

            epoch_sample_indices = sample_indices[epoch * self.mlm_batch_size:(epoch + 1) * self.mlm_batch_size]
            for idx in epoch_sample_indices:
                # Apply masking to the token ids
                input_chunk, masked_chunk, target_chunk, is_next_chunk = self.predict_dataset[idx]
                batch_chunks = self.add_to_batch(batch_chunks, input_chunk)
                batch_masked_chunks = self.add_to_batch(batch_masked_chunks, masked_chunk)
                batch_next_chunks = self.add_to_batch(batch_next_chunks, target_chunk)
                batch_is_next_chunk.append(is_next_chunk)

            # Convert the batch to tensors
            batch_chunks = batch_chunks.to(self.device)
            batch_masked_chunks = batch_masked_chunks.to(self.device)
            batch_next_chunks = batch_next_chunks.to(self.device)

            # Convert batch_is_next_chunk to a tensor
            batch_is_next_chunk = torch.tensor(batch_is_next_chunk, dtype=torch.float32).to(self.device)

            batch_masked_logits, batch_next_logits = self.model(batch_chunks, batch_masked_chunks, batch_next_chunks)

            # Calculate the loss
            masked_loss = self.calc_loss(batch_masked_logits, batch_chunks, loss_function)
            predict_loss = loss_function(
                batch_next_logits[:, 1].unsqueeze(1),  # Use the True prediction logits and reshape
                batch_is_next_chunk.unsqueeze(1),  # Reshape as a tensor of shape (batch_size, 1)
            )
            loss = masked_loss + predict_loss

            loss.backward()
            optimizer.step()

            # Update the total loss and number of batches
            total_loss += loss.item()
            num_batches += 1

            # Calculate the average loss over all batches
            avg_loss = total_loss / num_batches

            # Use pd.concat to append the new loss data to the existing loss data
            new_loss_df = pd.concat([
                new_loss_df, pd.DataFrame([[self.run_code, epoch + 1, avg_loss]], columns=["run_code", "epoch", "loss"])
            ], ignore_index=True)

        # Save loss data to a file
        if os.path.exists("loss.csv"):
            loss_df = pd.concat([pd.read_csv("loss.csv"), new_loss_df], ignore_index=True)
        else:
            loss_df = new_loss_df
        loss_df.to_csv("loss.csv", index=False)

        # Make sure the models directory exists
        if not os.path.exists(os.path.join("data", "models")):
            os.mkdir(os.path.join("data", "models"))

        # Save the model
        torch.save(
            self.model.state_dict(),
            os.path.join("data", "models", f"mlm_{self.run_code}.pt")
        )

    def prepare_for_quantization(self):
        """Prepare a model for quantization"""

        quant_config = torch.quantization.QConfig(
            activation=Int8MinMaxObserver, weight=torch.quantization.default_weight_observer
        )

        def apply_qconfig(module):
            if isinstance(module, nn.Linear):
                module.qconfig = quant_config
                torch.quantization.prepare_qat(module, inplace=True)

        self.model.apply(apply_qconfig)

    def load_model(self, run_code, vocab_size):
        """Load a trained model"""

        # Get configuration
        run_config_df = pd.read_csv("runs.csv")
        run_config = run_config_df[run_config_df["run_code"] == run_code].iloc[0]

        self.embedding_size = run_config["embedding_size"]
        self.mlm_epochs = run_config["epochs"]
        self.mlm_batch_size = run_config["batch_size"]
        self.mlm_lr = run_config["lr"]

        # Load the model
        self.model = DocumentEmbedder(
            vocab_size=vocab_size,
            embedding_size=run_config["embedding_size"],
            num_heads=run_config["num_heads"],
            dim_feedforward=run_config["dim_feedforward"],
            num_layers=run_config["num_layers"],
        )

        print("Preparing the model for quantization ...")
        self.prepare_for_quantization()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Load the model state dictionary
        self.model.load_state_dict(torch.load(os.path.join("data", "models", f"mlm_{run_code}.pt")))


def convert_to_quantized(model):
    """Convert a trained model to a quantized model"""
    model.cpu()
    model.eval()
    torch.quantization.convert(model, inplace=True)
    return model


def chunk_data(chunk_size):
    """
    Break each document in data/youtube/raw into 1024 token chunks.
    Divide into train/test sets, and save our respective datasets to the train/test directories.
    """

    train_dir = os.path.join("data", f"train-{chunk_size}")
    test_dir = os.path.join("data", f"test-{chunk_size}")

    # Create directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Empty directories
    for filename in os.listdir(train_dir):
        os.remove(os.path.join(train_dir, filename))
    for filename in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, filename))

    # Iterate over each document, with tqdm to show progress
    for filename in tqdm(os.listdir(os.path.join("data", "youtube", "raw"))):
        # Load the document
        with open(os.path.join("data", "youtube", "raw", filename), "r") as f:
            doc = f.read()

        # Tokenize the document, and get the integer token ids
        doc_tokens = [EMBED_VOCAB.stoi[token] for token in word_tokenize(doc) if token in EMBED_VOCAB.stoi]

        # Generate random number to determine if this file is in the train or test set
        if random() < 0.8:
            chunk_dir = train_dir
        else:
            chunk_dir = test_dir

        # Chunk the document into CHUNK_SIZE-1 token chunks
        for i in range(0, len(doc_tokens), chunk_size):

            # Convert tokens to embeddings
            embeddings = torch.tensor([embedding for embedding in doc_tokens[i:i+chunk_size]])
            # Pad the chunk with PAD_TOKEN_ID if it is smaller than CHUNK_SIZE
            embeddings = torch.nn.functional.pad(
                embeddings, (0, chunk_size - len(embeddings)), "constant", PAD_TOKEN_ID
            )

            # Save the chunk
            torch.save(embeddings, os.path.join(chunk_dir, f"{filename}_{i//chunk_size:04d}.pt"))


def record_run(run_code, chunk_size, batch_size, epochs, embedding_size, num_heads, dim_feedforward, num_layers, lr):
    # Load data from previous runs
    if os.path.exists("runs.csv"):
        run_df = pd.read_csv("runs.csv")
    else:
        run_df = pd.DataFrame(columns=[
            "run_code", "chunk_size", "batch_size", "epochs", "embedding_size", "num_heads", "dim_feedforward",
            "num_layers", "lr"
        ])

    # Record the run hyper-parameters for this run
    run_df = pd.concat([
        run_df, pd.DataFrame({
            "run_code": [run_code], "chunk_size": [chunk_size], "batch_size": [batch_size],
            "epochs": [epochs], "embedding_size": [embedding_size], "num_heads": [num_heads],
            "dim_feedforward": [dim_feedforward], "num_layers": [num_layers], "lr": [lr]
        })
    ])
    run_df.to_csv("runs.csv", index=False)


def load_run_config(run_code):
    # Load data from previous runs
    if os.path.exists("runs.csv"):
        run_df = pd.read_csv("runs.csv")
    else:
        raise FileNotFoundError("No runs.csv file found.")

    # Get the hyper-parameters for this run
    run_config = run_df[run_df["run_code"] == run_code].iloc[0]
    return run_config


if __name__ == "__main__":
    """
    Run with --train to train the model, and --chunk to chunk the data.
    --train, --reinforce and --chunk can be used together.
    --embed-size, --num-heads, --dim-feedforward, --num-layers, --lr, and --epochs can be used to change the model
    """

    parser = argparse.ArgumentParser()
    # Run modes
    parser.add_argument("--chunk", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--reinforce", action="store_true")

    # Model hyper-parameters
    parser.add_argument("--chunk-size", type=int, default=64)

    # Model training hyper-parameters
    train_group = parser.add_argument_group("model training")
    train_group.add_argument("--batch-size", type=int, default=8)
    train_group.add_argument("--dim-feedforward", type=int, default=512)
    train_group.add_argument("--epochs", type=int, default=9000)
    train_group.add_argument("--embedding-size", type=int, default=128)
    train_group.add_argument("--num-heads", type=int, default=4)
    train_group.add_argument("--num-layers", type=int, default=2)
    train_group.add_argument("--lr", type=float, default=1e-4)

    # Run code to load a trained model
    parser.add_argument("--run-code", type=str, default=None)
    pargs = parser.parse_args()

    # Chunk the data
    if pargs.chunk:
        print("Chunking data...")
        chunk_data(pargs.chunk_size)

    # Train the model
    trainer = None
    if pargs.train:
        print("Training model...")
        trainer = DocumentEmbeddingTrainer(
            chunk_size=pargs.chunk_size, embedding_size=pargs.embedding_size
        )
        trainer.init_model(
            batch_size=pargs.batch_size, num_epochs=pargs.epochs, num_heads=pargs.num_heads,
            dim_feedforward=pargs.dim_feedforward, num_layers=pargs.num_layers, lr=pargs.lr
        )
        trainer.train()

        record_run(
            trainer.run_code, pargs.chunk_size, pargs.batch_size, pargs.epochs, pargs.embedding_size, pargs.num_heads,
            pargs.dim_feedforward, pargs.num_layers, pargs.lr
        )
