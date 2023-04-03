"""
Usage:
    Break our documents into chunks of 1024 words each, and save them as PyTorch tensors:
    python doc_embed_torch.py --chunk

    Train the model:
    python doc_embed_torch.py --train
"""

import argparse
import os
import string
from random import random, shuffle, choice, sample

import pandas as pd
import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset, DataLoader
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


class MaskedTextDataset(Dataset):
    def __init__(self, dataset, mask_prob=0.15):
        self.dataset = dataset  # Original dataset containing token ids
        self.mask_prob = mask_prob  # Probability of masking a token

    def mask_tokens(self, token_ids):
        masked_token_ids = token_ids.clone()  # Create a copy of token ids
        target = []  # Initialize a list to store the original token ids

        # Iterate through the token ids and mask tokens with a probability of self.mask_prob
        for i, token_id in enumerate(token_ids):
            if random() < self.mask_prob:
                masked_token_ids[i] = MASK_TOKEN_ID  # Replace the token id with mask_token_id
                target.append(token_id.item())  # Store the original token id in target
            else:
                target.append(token_id.item())  # Store the original token id in target

        return masked_token_ids, target

    def __len__(self):
        return len(self.dataset)  # Return the total number of samples in the dataset

    def __getitem__(self, idx):
        token_ids = self.dataset[idx]  # Get the token ids for the current sample
        masked_token_ids, target_ids = self.mask_tokens(token_ids)  # Mask the tokens in the current sample
        return masked_token_ids, torch.tensor(target_ids)


class DocumentMLMEmbedder(nn.Module):
    """This model takes a document as input and outputs token-level predictions."""

    def __init__(self, vocab_size, embedding_size, num_heads, dim_feedforward, num_layers):
        super(DocumentMLMEmbedder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            activation='relu',
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
        )
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, return_doc_embedding=False):
        """
        Predict token-level embeddings from a document
        x is of shape (batch_size, seq_len)
        """

        # Embed the input sequence
        embedded = self.embedding(x)  # shape (batch_size, seq_len, embedding_size)
        # Encode the input sequence using the transformer's encoder and decoder
        encoded = self.transformer(embedded, embedded)  # shape (batch_size, seq_len, embedding_size)

        if return_doc_embedding:
            # Use the output of the last token in the sequence
            doc_embedding = encoded[-1]  # shape (batch_size, embedding_size)
            return doc_embedding

        # Output token-level predictions
        logits = self.fc(encoded)  # shape (batch_size, seq_len, vocab_size)

        return logits


class DocumentCleanEmbedder(nn.Module):
    """This model takes a document as input and outputs a document-level embedding."""

    def __init__(self, vocab_size, embedding_size, num_heads, dim_feedforward, num_layers):
        super(DocumentCleanEmbedder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            activation='relu',
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
        )

    def forward(self, x):
        """
        Predict document-level embeddings from a document
        x is of shape (batch_size, seq_len)
        """

        # Embed the input sequence
        embedded = self.embedding(x)  # shape (batch_size, seq_len, embedding_size)
        # Encode the input sequence using the transformer's encoder and decoder
        encoded = self.transformer(embedded, embedded)  # shape (batch_size, seq_len, embedding_size)
        # Get the output embedding of the [CLS] token (assumed to be at position 0)
        doc_embedding = encoded[:, 0, :]  # shape (batch_size, embedding_size)

        return doc_embedding


class DocumentEmbeddingAdversary(torch.nn.Module):
    """This model takes a document embedding as input and outputs a cluster label."""

    def __init__(self, embedding_size, max_clusters=20):
        super().__init__()
        # First linear layer: takes input of size embed_dim and maps to embedding_size // 2
        self.layer1 = torch.nn.Linear(embedding_size, embedding_size // 2)
        # Second linear layer: takes input of size (embedding_size // 2) and maps it to max_clusters
        self.cluster_layers = torch.nn.ModuleList([
            torch.nn.Linear(embedding_size // 2, 1) for _ in range(max_clusters)
        ])

    def forward(self, x, num_clusters):
        """Predict cluster label from a document embedding"""

        # Pass input x through the first layer and apply ReLU activation
        x_relu = torch.relu(self.layer1(x))

        # Pass the output of the first layer through the second layer
        logits = torch.cat([layer(x_relu) for layer in self.cluster_layers[:num_clusters]], dim=1)
        # Return logits representing the probability of each cluster
        return logits


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

        self.train_dataset = DocumentDataset(os.path.join("data", f"train-{self.chunk_size}"))
        self.test_dataset = DocumentDataset(os.path.join("data", f"test-{self.chunk_size}"))

        # Prepare dataset, dataloader, and masking for MLM
        print("Preparing the masked dataset ...")
        self.mask_prob = 0.15
        self.masked_dataset = MaskedTextDataset(self.train_dataset, self.mask_prob)
        self.masked_dataloader = DataLoader(self.masked_dataset, batch_size=1, shuffle=True)
        print("Done preparing the masked dataset.")

    def init_mlm(self, batch_size, num_epochs, num_heads, dim_feedforward, num_layers, lr):
        # Create the DocumentMLMEmbedder model
        self.mlm_lr = lr
        self.mlm_epochs = num_epochs
        self.mlm_batch_size = batch_size

        print("Creating the MLM model ...")
        self.model = DocumentMLMEmbedder(
            vocab_size=VOCAB_SIZE,
            embedding_size=self.embedding_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
        ).to(self.device)

    def calc_mlm_loss(self, input_ids, labels, loss_function):
        # Forward pass to get the logits
        logits = self.model(input_ids)
        # Calculate the loss
        batch_size, seq_len = input_ids.shape
        logits_view = logits.view(batch_size * seq_len, -1)
        labels_view = labels.view(batch_size * seq_len)
        loss = loss_function(logits_view, labels_view)
        return loss

    def train_mlm(self, epoch_size=8):
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Create the optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.mlm_lr)
        # Ignore the loss for masked tokens
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

        for i in range(epoch_size):
            for masked_batch in self.masked_dataloader:
                input_ids, labels = masked_batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                # Clear the gradients before each forward pass
                optimizer.zero_grad()
                # Forward pass to get the logits
                loss = self.calc_mlm_loss(input_ids, labels, loss_function)
                # Calculate the gradients of the loss with respect to all the parameters
                loss.backward()
                # Make a step in the optimizer to update the parameters
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= len(self.masked_dataloader):
                    break

            if num_batches >= len(self.masked_dataloader):
                break

        return total_loss / num_batches

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

        # Create the optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.mlm_lr)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Read the loss from a file if it exists
        new_loss_df = pd.DataFrame(columns=["run_code", "epoch", "loss"])

        # Train the model
        for epoch in tqdm(range(self.mlm_epochs)):
            # Sample [self.mlm_batch_size] random documents from the training dataset
            sample_indices = sample(range(len(self.train_dataset)), self.mlm_batch_size)

            total_loss = 0
            num_batches = 0

            for idx in sample_indices:
                optimizer.zero_grad()

                # Get the token ids for the current document
                token_ids = self.train_dataset[idx]

                # Apply masking to the token ids
                masked_token_ids_tensor, masked_target_ids = self.masked_dataset.mask_tokens(token_ids)
                # And convert target IDs to tensors
                target_ids_tensor = torch.tensor(masked_target_ids).unsqueeze(0).to(self.device)

                # Forward pass through the model
                logits = self.model(masked_token_ids_tensor.unsqueeze(0).to(self.device))

                # Calculate the loss
                loss = loss_function(logits.squeeze(0), target_ids_tensor.squeeze(0))
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

    def train_adversary(self, pseudo_labels, num_topics, epoch_size=8):
        """
        Train the document embedding adversary

        :param pseudo_labels: a dictionary of document ids and their corresponding pseudo labels
        :param num_topics: the number of topics to use for training
        :param epoch_size: the number of batches to train per epoch

        :return: the average loss per batch
        """

        # Set the model to training mode
        self.adversary.train()
        total_loss = 0
        num_batches = 0

        # Create the optimizer and loss function
        optimizer = torch.optim.Adam(self.adversary.parameters(), lr=self.adversary_lr)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

        for idx, pseudo_label in pseudo_labels.items():
            input_ids = self.train_dataset[idx]
            # Move the batch and label to the device
            input_ids = input_ids.to(self.device)
            embedding = self.model(input_ids, return_doc_embedding=True)
            pseudo_label = torch.tensor(pseudo_label).to(self.device).unsqueeze(0)

            # Zero out the gradients
            optimizer.zero_grad()
            # Generate the topic logits
            logits = self.adversary(embedding.unsqueeze(0), num_topics)
            # Calculate the loss comparing the logits to the label
            loss = loss_function(logits, pseudo_label)
            # Back-propagate the loss
            loss.backward()
            # Update the model parameters
            optimizer.step()

            # Update the total loss and number of batches
            total_loss += loss.item()
            num_batches += 1

            # Stop training if the number of batches exceeds the limit
            if epoch_size and num_batches >= epoch_size:
                break

        # Return the average loss
        return total_loss / num_batches

    def train_combined(self, alpha):
        """Reinforcement learning to train the DocumentMLMEmbedder model"""

        # Set both the model and the adversary to training mode
        self.model.train()
        self.adversary.train()

        # Create the optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.mlm_lr)
        # Ignore the loss for outliers and masked tokens
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

        print("Combined training...")
        optimizer.zero_grad()

        # Create a list of indices for the training dataset
        sample_indices = sample(range(len(self.train_dataset)), self.adversary_batch_size)
        # Get token_ids for the sample_indices of this indices as a tensor
        # Convert the list of token id tensors to a tensor of token id tensors
        input_ids = torch.stack([self.train_dataset[idx] for idx in sample_indices])

        # Get masked input ids and labels for each document in the batch and concatenate them in tensors
        masked_input_ids = None
        batch_labels = None
        for idx, doc_token_ids in enumerate(input_ids):
            # Apply masking to the token ids
            doc_masked_input_ids, doc_labels = self.masked_dataset.mask_tokens(doc_token_ids)
            # Convert the labels to tensors
            doc_labels = torch.tensor(doc_labels).to(self.device)

            # If this is the first document, create the masked_input_ids and batch_labels tensors
            if masked_input_ids is None:
                masked_input_ids = doc_masked_input_ids.unsqueeze(0)
                batch_labels = doc_labels.unsqueeze(0)
            # Otherwise, concatenate the masked_input_ids and batch_labels tensors
            else:
                masked_input_ids = torch.concat([masked_input_ids, doc_masked_input_ids.unsqueeze(0)])
                batch_labels = torch.concat([batch_labels, doc_labels.unsqueeze(0)])

        # Move the masked inputs and labels to the device
        masked_input_ids = masked_input_ids.to(self.device)
        # Make sure the labels are a tensor
        if not isinstance(batch_labels, torch.Tensor):
            batch_labels = torch.tensor(batch_labels).to(self.device)

        embeddings = self.model(masked_input_ids, return_doc_embedding=True)

        # Compute the MLM loss
        mlm_loss = self.calc_mlm_loss(masked_input_ids, batch_labels, loss_function)

        # Generate the document embeddings and pseudo-labels
        pseudo_labels = self.generate_pseudo_labels(embeddings=embeddings, sample_indices=sample_indices)
        # Get the number of topics from the pseudo-labels
        topic_set = set(pseudo_labels.values())
        num_topics = len(topic_set) - 1 if -1 in topic_set else len(topic_set)

        # Compute the document embedding adversary loss
        if num_topics == 0:
            return None

        adversary_logits = self.adversary(embeddings, num_topics)
        try:
            adversary_targets = torch.tensor(
                [pseudo_labels[idx.item()] for idx in input_ids[:, 0]], dtype=torch.long, device=self.device
            )
        except KeyError as ke:
            raise KeyError(f"KeyError: {ke} in {pseudo_labels.keys()}")
        adversary_loss = loss_function(adversary_logits, adversary_targets)

        # Compute the combined loss
        combined_loss = mlm_loss + alpha * adversary_loss

        # Back-propagate the combined loss
        combined_loss.backward()

        # Update the model parameters
        optimizer.step()

        # Update the total loss and number of batches
        print(f"MLM loss: {mlm_loss.item()} Adversary loss: {adversary_loss.item()} Combined: {combined_loss.item()}")
        return combined_loss.item()

    def generate_pseudo_labels(self, sample_size=64, embeddings=None, sample_indices=None):
        """
        Generate pseudo-labels for the documents using DBScan.
        There are two ways to generate pseudo-labels:
            1. Generate document embeddings for a random subset of the dataset
            2. Use the document embeddings from the current batch. This is used for reinforcement learning.

        For the first method, supply only the sample_size parameter.
        For the second method, supply the embeddings and sample_indices parameters.

        :param sample_size: the number of documents to use for generating pseudo-labels
        :param embeddings: a tensor of document embeddings
        :param sample_indices: a list of indices for the training dataset
        :return: a dictionary of document ids and their corresponding pseudo labels
        """

        # Change the model from training to evaluation mode
        self.model.eval()

        if embeddings is None:
            # Generate document embeddings for a random subset of the dataset
            embeddings = torch.Tensor()
            dataset_size = len(self.train_dataset)

            if sample_size is None or sample_size > dataset_size:
                sample_size = dataset_size

            # Choose a random sample of indices from the dataset
            sample_indices = sample(range(dataset_size), sample_size)

            # Disable gradient calculation to speed up inference
            with torch.no_grad():
                print("Generating document embeddings...")
                for idx in tqdm(sample_indices):
                    batch = self.train_dataset[idx]
                    # Generate document embeddings for the documents in the batch
                    doc_embedding = self.model(batch.to(self.device), return_doc_embedding=True)
                    embeddings = torch.cat([embeddings, doc_embedding.unsqueeze(0)])

        # Use DBScan to cluster the document embeddings
        # This will generate a list of cluster labels for each document
        # The label -1 indicates that the document is an outlier
        db = DBSCAN(eps=20, min_samples=3)
        # Fit the DBScan model to the document embeddings
        print("Clustering document embeddings...")
        db.fit(embeddings.detach().numpy())
        # create dict of cluster labels
        doc_labels = {idx: label for idx, label in zip(sample_indices, db.labels_)}

        # Return the document labels
        return doc_labels

    def load_mlm(self, run_code, vocab_size):
        """Load the trained model"""

        # Get configuration
        run_config_df = pd.read_csv("runs.csv")
        run_config = run_config_df[run_config_df["run_code"] == run_code].iloc[0]

        self.embedding_size = run_config["embedding_size"]
        self.mlm_epochs = run_config["epochs"]
        self.mlm_batch_size = run_config["batch_size"]
        self.mlm_lr = run_config["lr"]

        # Load the model
        self.model = DocumentMLMEmbedder(
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

    def init_reinforce(self, embedding_size, max_topics, adv_epochs=100, adv_batch_size=64, adv_lr=1e-3,
                       comb_epochs=100):
        """Initialize the document embedding adversary and reinforcement learning parameters"""

        self.adversary = DocumentEmbeddingAdversary(embedding_size, max_topics)
        self.adversary.to(self.device)
        self.adversary_epochs = adv_epochs
        self.adversary_batch_size = adv_batch_size
        self.adversary_lr = adv_lr
        self.combined_epochs = comb_epochs

    def reinforce(self, run_code, epochs=100):
        """
        Use reinforcement learning to train DocumentMLMEmbedder to model topics.
        :param run_code: The run code for the MLM-trained DocumentMLMEmbedder model to use.
        :param epochs: The number of epochs to train the adversary.
        """

        # Load the chunk size from runs.csv
        run_log_df = pd.read_csv("runs.csv")
        run_log_df = run_log_df[run_log_df["run_code"] == run_code]
        self.chunk_size = run_log_df["chunk_size"].values[0]
        self.embedding_size = run_log_df["embedding_size"].values[0]

        # Load the pre-trained DocumentEmbeddingTransformer model from the run code
        print("Loading the DocumentMLMEmbedder model ...")
        self.load_mlm(run_code, VOCAB_SIZE)

        epoch = 0
        while epoch < epochs:
            print(f"Starting Reinforcement Epoch {epoch + 1} ...")

            # 2. Generate pseudo-labels for the documents using DBScan
            print("Generating pseudo-labels ...")
            pseudo_labels = self.generate_pseudo_labels(sample_size=256)
            # Get the number of topics from the pseudo-labels. The indices and labels are zipped together
            topic_set = set(pseudo_labels.values())
            num_topics = len(topic_set) - 1 if -1 in topic_set else len(topic_set)

            # Only train the adversary on the first iteration
            if epoch == 0:
                # 3. Train the DocumentEmbeddingAdversary model to predict the pseudo-labels for the documents.
                print("Training the DocumentEmbeddingAdversary model ...")
                # Train the DocumentEmbeddingAdversary model using the pseudo-labels as the target
                self.train_adversary(pseudo_labels, num_topics=num_topics)

            # 4. Update the DocumentMLMEmbedder using a combined loss that includes the original loss and an
            # adversarial loss based on the DocumentEmbeddingAdversary's predictions.
            print("Updating the DocumentMLMEmbedder model ...")
            combined_loss = self.train_combined(alpha=0.5)
            if combined_loss is None:
                print("Combined loss is None. Skipping this epoch.")
                continue

            # 5. Update the MLM model parameters with reference to the loss

            epoch += 1
            print(f"Combined loss: {combined_loss}")


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
    for filename in tqdm(os.listdir("data/youtube/raw")):
        # Load the document
        with open(f"data/youtube/raw/{filename}", "r") as f:
            doc = f.read()

        # Tokenize the document, and get the integer token ids
        doc_embeddings = [EMBED_VOCAB.stoi[token] for token in word_tokenize(doc) if token in EMBED_VOCAB.stoi]

        # Chunk the document into CHUNK_SIZE-1 token chunks
        for i in range(0, len(doc_embeddings), chunk_size-1):
            # Generate random number to determine if this chunk is in the train or test set
            if random() < 0.8:
                chunk_dir = train_dir
            else:
                chunk_dir = test_dir

            # Convert tokens to embeddings
            embeddings = torch.tensor([CLS_TOKEN_ID] + [embedding for embedding in doc_embeddings[i:i+chunk_size-1]])
            # Pad the chunk with PAD_TOKEN_ID if it is smaller than CHUNK_SIZE
            embeddings = torch.nn.functional.pad(
                embeddings, (0, chunk_size - len(embeddings)), "constant", PAD_TOKEN_ID
            )

            # Save the chunk
            torch.save(embeddings, os.path.join(chunk_dir, f"{filename}_{i}.pt"))


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
    parser.add_argument("--chunk-size", type=int, default=128)

    # MLM training hyper-parameters
    train_group = parser.add_argument_group("mlm training")
    train_group.add_argument("--batch-size", type=int, default=8)
    train_group.add_argument("--dim-feedforward", type=int, default=512)
    train_group.add_argument("--epochs", type=int, default=100)
    train_group.add_argument("--embedding-size", type=int, default=128)
    train_group.add_argument("--num-heads", type=int, default=4)
    train_group.add_argument("--num-layers", type=int, default=2)
    train_group.add_argument("--lr", type=float, default=1e-4)

    # Reinforcement learning hyper-parameters
    reinforce_group = parser.add_argument_group("reinforcement")
    reinforce_group.add_argument("--max-topics", type=int, default=20)
    reinforce_group.add_argument("--adv-epochs", type=int, default=10)
    reinforce_group.add_argument("--adv-batch-size", type=int, default=16)
    reinforce_group.add_argument("--adv-lr", type=float, default=1e-3)
    reinforce_group.add_argument("--comb-epochs", type=int, default=100)

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
        trainer.init_mlm(
            batch_size=pargs.batch_size, num_epochs=pargs.epochs, num_heads=pargs.num_heads,
            dim_feedforward=pargs.dim_feedforward, num_layers=pargs.num_layers, lr=pargs.lr
        )
        trainer.train()

        record_run(
            trainer.run_code, pargs.chunk_size, pargs.batch_size, pargs.epochs, pargs.embedding_size, pargs.num_heads,
            pargs.dim_feedforward, pargs.num_layers, pargs.lr
        )

    # Reinforce the model
    if pargs.reinforce:
        print("Reinforcing model...")

        if pargs.run_code is None:
            raise ValueError("Must specify run code to reinforce model")

        # Load the run config
        my_run_config = load_run_config(pargs.run_code)

        if trainer is None:
            trainer = DocumentEmbeddingTrainer(run_code=pargs.run_code)
            trainer.load_mlm(run_code=pargs.run_code, vocab_size=VOCAB_SIZE)
        trainer.init_reinforce(
            embedding_size=my_run_config.embedding_size, max_topics=pargs.max_topics, adv_epochs=pargs.adv_epochs,
            adv_batch_size=pargs.adv_batch_size, adv_lr=pargs.adv_lr, comb_epochs=pargs.comb_epochs
        )
        trainer.reinforce(run_code=pargs.run_code)
