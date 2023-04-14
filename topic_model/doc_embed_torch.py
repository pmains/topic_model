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
import gc
import os
import string
from random import random, shuffle, choice, sample

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from memory_profiler import profile
from nltk.tokenize import word_tokenize
from sklearn.metrics import r2_score, f1_score, accuracy_score
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
        self.file_list = [doc for doc in os.listdir(self.folder_path) if doc.endswith(".pt")]
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


class PredictChunkDataset(Dataset):
    def __init__(self, dir_path, mask_prob=0.15):
        """
        Files in our directory will have a name in format "{channel}-{video_id}_{chunk_id}.pt"
        The chunk_id values are sequential, so we can use them to sort the files.
        Files are only considered sequential if they belong to the same channel and video_id
        These pairs are generated during the --chunk preprocessing step and stored in a CSV file.

        :param dir_path: Path to the directory containing the dataset
        """

        self.dir_path = dir_path  # Path to the directory containing the dataset
        self.chunk_list = []
        self.mask_prob = mask_prob

        # Get a list of all files in the directory and create a dataframe
        print("Grouping files by channel and video_id ...")
        file_list = [file_name for file_name in os.listdir(self.dir_path) if file_name.endswith(".pt")]

        folder, chunk_size = self.dir_path.split("-")
        data_type = folder.split(os.sep)[-1]
        is_train = data_type == 'train'

        chunk_df = pd.read_csv(os.path.join("data", f"sequential-chunks-{chunk_size}.csv"))
        chunk_df = chunk_df[chunk_df['is_train'] == is_train]

        for _, chunk_pair in chunk_df.iterrows():
            video_id = chunk_pair.video_id
            # File names in format "[file name]._[chunk id].pt
            chunk_a_id = chunk_pair.chunk_a
            chunk_b_id = chunk_pair.chunk_a
            chunk_a_file = f"{video_id}_{chunk_a_id:04d}.pt"
            if random() < .5:
                chunk_b_file = f"{video_id}_{chunk_b_id:04d}.pt"
                is_next_chunk = True
            else:
                chunk_b_file = sample(file_list, 1)[0]
                is_next_chunk = False

            self.chunk_list.append((chunk_a_file, chunk_b_file, is_next_chunk))

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


class EmbeddedDocumentDataset(Dataset):
    """Dataset that return (document, embedding) for a given chunk size and embedder"""
    def __init__(self, chunk_size, embedder):
        self.chunk_size = chunk_size
        self.embedder = embedder

        self.file_list = []
        train_dir = os.path.join("data", f"train-{chunk_size}")
        test_dir = os.path.join("data", f"test-{chunk_size}")
        # Get a list of all token ID files in the directory
        for dir_path in (train_dir, test_dir):
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".pt"):
                    self.file_list.append(os.path.join(dir_path, file_name))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get the token ids for the file at the current index
        token_file_path = self.file_list[idx]
        # Make sure tensor is of shape (1, chunk_size)
        token_ids = torch.load(token_file_path).unsqueeze(0)
        # Get the document embedding for the token ids
        doc_embedding = self.embedder(token_ids, return_doc_embedding=True)
        # Get the corresponding text for the token ids
        text_file_path = token_file_path.replace(".pt", ".txt")
        with open(text_file_path, "r") as text_file:
            document = text_file.read()

        return doc_embedding, document


class DocumentMLMEmbedder(nn.Module):
    """Embeds documents using a masked language model for training"""

    def __init__(self, vocab_size, embedding_size, num_heads, dim_feedforward, num_layers):
        super(DocumentMLMEmbedder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.transformer = nn.Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            activation='gelu',
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
        )
        self.fc = nn.Linear(embedding_size, vocab_size)
        self.idf_weights = None

    def forward(self, x, return_doc_embedding=False):
        embedded = self.embedding(x)
        encoded = self.transformer(embedded, embedded)

        if return_doc_embedding:
            if self.idf_weights is None:
                self.idf_weights = np.load(os.path.join("data", "idf_vector.npy"))

            # Create a tensor with the same shape as encoded and fill it with the corresponding IDF values
            batch_embeddings = None
            for encoded_doc, doc in zip(encoded, x):
                idf_list = [self.idf_weights[token_id] for token_id in doc]
                idf_tensor = torch.tensor(idf_list).unsqueeze(1).repeat(1, encoded_doc.shape[1])
                # Take the mean of the IDF-weighted encoded sequence to get a document-level embedding
                doc_embedding = (encoded_doc * idf_tensor).sum(dim=1).squeeze(0)
                batch_embeddings = add_to_batch(batch_embeddings, doc_embedding)

            return batch_embeddings

        # Predict the masked tokens
        logits = self.fc(encoded)
        return logits


class DocumentDualEmbedder(nn.Module):
    """Embeds documents using a masked language model for training"""

    def __init__(self, vocab_size, embedding_size, num_heads, dim_feedforward, num_layers):
        super(DocumentDualEmbedder, self).__init__()

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
            nn.Linear(embedding_size*8, 2),
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

        # Delete unnecessary tensors to save memory
        del embedded_chunk, encoded_chunk, embedded_masked_chunk, encoded_masked_chunk, embedded_next_chunk, \
            encoded_next_chunk
        gc.collect()

        return masked_logits, next_chunk_prob

    def next_chunk_prediction(self, doc_embedding, next_chunk_embedding):
        concatenated_embeddings = torch.cat((doc_embedding, next_chunk_embedding), dim=1)
        next_chunk_prob = self.next_chunk_classifier(concatenated_embeddings)

        # Delete unnecessary tensors to save memory
        del concatenated_embeddings
        gc.collect()

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
            mean_embedding = (encoding_vec * idf_weights).sum(dim=0)
            max_embedding = encoding_vec.max(dim=0)[0]
            min_embedding = encoding_vec.min(dim=0)[0]
            std_embedding = encoding_vec.std(dim=0)
            # Concatenate the mean, max, min, and std embeddings
            # doc_embedding_vec = torch.cat((mean_embedding, max_embedding)).unsqueeze(0)
            doc_embedding_vec = torch.cat((mean_embedding, max_embedding, min_embedding, std_embedding)).unsqueeze(0)

            # Add the document embedding to the doc_embeddings tensor
            if doc_embeddings is None:
                doc_embeddings = doc_embedding_vec
            else:
                doc_embeddings = torch.cat((doc_embeddings, doc_embedding_vec))

        # Delete unnecessary tensors to save memory
        del idf_weights, mean_embedding, max_embedding, min_embedding, std_embedding, doc_embedding_vec
        gc.collect()

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

    def __init__(self, chunk_size=None, embedding_size=None, run_code=None, model_type=None, is_test=False):
        if run_code is None:
            self.run_code = gen_run_code()
            print(f"{model_type.upper()} model {self.run_code}")
            self.chunk_size = chunk_size
            self.embedding_size = embedding_size
        else:
            self.run_code = run_code
            run_config = load_run_config(run_code)
            self.chunk_size = run_config.chunk_size
            self.embedding_size = run_config.embedding_size

        self.model = None

        # Hyper-parameters
        self.lr = None
        self.epochs = None
        self.batch_size = None

        # Train the DocumentEmbeddingTransformer to generate document embeddings.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if is_test:
            self.chunk_dir = os.path.join("data", f"test-{self.chunk_size}")
        else:
            self.chunk_dir = os.path.join("data", f"train-{self.chunk_size}")

        self.doc_dataset = DocumentDataset(self.chunk_dir)
        self.masked_dataset = None
        self.predict_dataset = None
        self.model_type = model_type

    def init_model(self, batch_size, num_epochs, num_heads, dim_feedforward, num_layers, lr):
        # Create the DocumentDualEmbedder model
        self.lr = lr
        self.epochs = num_epochs
        self.batch_size = batch_size

        if self.model_type == "dual":
            self.predict_dataset = PredictChunkDataset(self.chunk_dir)

            print("Creating the Dual model ...")
            self.model = DocumentDualEmbedder(
                vocab_size=VOCAB_SIZE,
                embedding_size=self.embedding_size,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                num_layers=num_layers,
            ).to(self.device)
        else:
            self.masked_dataset = MaskedTextDataset(self.doc_dataset)

            print("Creating the MLM model ...")
            self.model = DocumentMLMEmbedder(
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
        del logits_view, labels_view
        gc.collect()
        return loss

    @profile
    def train_dual(self):
        """Train the DocumentDualEmbedder model"""

        if self.model is None:
            raise ValueError("The model has not been initialized. Call init_dual() or load_model() first.")

        # Prepare the model for quantization
        self.prepare_for_quantization()
        self.model.to(self.device)
        self.model.train()

        # Create the optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Read the loss from a file if it exists
        new_loss_df = pd.DataFrame(columns=["run_code", "epoch", "loss"])

        # Train the model
        print("Training the Dual model ...")
        dataloader = DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=True)

        # Determine the number of epochs to display in the progress bar
        num_epochs = min(self.epochs, len(dataloader)) if self.epochs is not None else len(dataloader)

        # Wrap the training loop with a tqdm iterator
        with tqdm(total=num_epochs, desc="Training", unit="epoch") as progress_bar:
            for batch_count, batch in enumerate(dataloader, 1):
                # Unpack the batch
                batch_chunk, batch_mask, batch_next, batch_is_next = batch

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass through the model
                batch_masked_logits, batch_next_matrix = self.model(
                    batch_chunk.to(self.device),
                    batch_mask.to(self.device),
                    batch_next.to(self.device)
                )

                # Calculate the loss
                masked_loss = self.calc_loss(batch_masked_logits, batch_chunk, loss_function)
                predict_loss = loss_function(
                    batch_next_matrix[:, 1],  # Use the True prediction logits and reshape
                    batch_is_next.float(),  # Reshape as a tensor of shape (batch_size, 1)
                )
                loss = masked_loss + predict_loss

                loss.backward()
                optimizer.step()

                # Use pd.concat to append the new loss data to the existing loss data
                new_loss_df = pd.concat([
                    new_loss_df,
                    pd.DataFrame([[self.run_code, batch_count + 1, loss]], columns=["run_code", "epoch", "loss"])
                ], ignore_index=True)

                # Update the progress bar
                progress_bar.update(1)
                # Empty the GPU cache
                torch.cuda.empty_cache()
                # Delete tensors to free up memory
                del batch, batch_chunk, batch_mask, batch_next, batch_is_next, batch_masked_logits, batch_next_matrix
                gc.collect()

                # Limit the number of epochs
                if self.epochs is not None and batch_count >= self.epochs:
                    break

        self.save_loss(new_loss_df)
        self.save_model()

    def train_mlm(self):
        """Train the DocumentMLMEmbedder model"""

        if self.model is None:
            raise ValueError("The model has not been initialized. Call init_mlm() or load_model() first.")

        # Prepare the model for quantization
        self.prepare_for_quantization()
        self.model.to(self.device)

        # Create the optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Read the loss from a file if it exists
        new_loss_df = pd.DataFrame(columns=["run_code", "epoch", "loss"])

        # Train the model
        print("Training the MLM model ...")

        # Load batches of the test dataset
        dataloader = DataLoader(self.masked_dataset, batch_size=self.batch_size, shuffle=True)

        # Determine the number of epochs to display in the progress bar
        num_epochs = min(self.epochs, len(dataloader)) if self.epochs != 0 else len(dataloader)

        # Wrap the training loop with a tqdm iterator
        with tqdm(total=num_epochs, desc="Training", unit="epoch") as progress_bar:
            for batch_count, batch in enumerate(dataloader, 1):
                # Unpack the MaskedTextDataset batch
                masked_token_ids, target_token_ids = batch

                # Forward pass through the model
                batch_logits = self.model(masked_token_ids.to(self.device)).to(self.device)

                # Calculate the loss
                batch_logits_flat = batch_logits.view(-1, batch_logits.shape[-1])
                batch_targets_flat = target_token_ids.view(-1)
                loss = loss_function(batch_logits_flat, batch_targets_flat)
                loss.backward()
                optimizer.step()

                # Use pd.concat to append the new loss data to the existing loss data
                new_loss_df = pd.concat([new_loss_df, pd.DataFrame(
                    [[self.run_code, batch_count + 1, loss.item()]], columns=["run_code", "epoch", "loss"]
                )], ignore_index=True)

                # Update the progress bar
                progress_bar.update(1)
                # Empty the GPU cache
                torch.cuda.empty_cache()

                if self.epochs is not None and batch_count > self.epochs:
                    break

        self.save_loss(new_loss_df)
        self.save_model()

    def validate(self):
        """Calculate the R2, F1 and Accuracy Score of our Model on the Test Dataset"""

        r2 = f1 = accuracy = 0

        batch_count = 0
        if self.model_type == "mlm":
            # Load batches of the test dataset
            dataloader = DataLoader(self.masked_dataset, batch_size=self.batch_size, shuffle=False)

            for batch_count, batch in enumerate(dataloader, 1):
                # Unpack the MaskedTextDataset batch
                masked_token_ids, target_token_ids = batch

                # Forward pass through the model
                masked_logits = self.model(masked_token_ids.to(self.device)).to(self.device)
                # Convert logits to predictions using argmax
                mask = masked_token_ids == MASK_TOKEN_ID
                masked_pred = torch.argmax(masked_logits, dim=2)[mask]
                masked_target_ids = target_token_ids[mask]

                # Calculate R1, F1 and Accuracy Score
                r2 += r2_score(masked_target_ids, masked_pred)
                f1 += f1_score(masked_target_ids, masked_pred, average="weighted")
                accuracy += accuracy_score(masked_target_ids, masked_pred)

                if batch_count % 100 == 0:
                    print(f"Batch {batch_count} of {len(dataloader)}")
        elif self.model_type == "dual":
            mask_r2 = mask_f1 = mask_accuracy = 0
            next_r2 = next_f1 = next_accuracy = 0

            # Load batches of the test dataset
            data_loader = DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False)

            batch_count = 0
            for batch_chunk, batch_mask, batch_next, batch_is_next in data_loader:
                # Forward pass through the model
                batch_masked_logits, batch_next_matrix = self.model(
                    batch_chunk.to(self.device),
                    batch_mask.to(self.device),
                    batch_next.to(self.device)
                )
                # Calculate R1, F1 and Accuracy Score
                batch_masked_pred = torch.argmax(batch_masked_logits, dim=2)
                batch_next_pred = batch_next_matrix[:, 1]

                mask = batch_mask == MASK_TOKEN_ID
                batch_masked_pred = batch_masked_pred[mask]
                batch_masked_target = batch_chunk[mask]

                mask_r2 += r2_score(batch_masked_target, batch_masked_pred)
                mask_f1 += f1_score(batch_masked_target, batch_masked_pred, average="weighted")
                mask_accuracy += accuracy_score(batch_masked_target, batch_masked_pred)

                # Detach the tensors from the graph and convert them to numpy arrays
                batch_is_next_float = batch_is_next.float().detach().cpu().numpy()
                batch_next_pred = batch_next_pred.detach().cpu().numpy()
                next_r2 += r2_score(batch_is_next_float, batch_next_pred)
                next_f1 += f1_score(batch_is_next_float, batch_next_pred.round(), average="weighted")
                next_accuracy += accuracy_score(batch_is_next_float, batch_next_pred.round())

                r2 = (mask_r2 + next_r2) / 2
                f1 = (mask_f1 + next_f1) / 2
                accuracy = (mask_accuracy + next_accuracy) / 2

                batch_count += 1
                if batch_count % 100 == 0:
                    print(f"Batch {batch_count} of {len(data_loader)}")
        else:
            raise ValueError("Invalid model type")

        if batch_count == 0:
            print("No batches found")
            return 0, 0, 0

        r2 /= batch_count
        f1 /= batch_count
        accuracy /= batch_count

        return r2, f1, accuracy

    @staticmethod
    def save_loss(new_loss_df):
        """Save the loss data to a file"""

        # Save loss data to a file
        if os.path.exists("loss.csv"):
            loss_df = pd.concat([pd.read_csv("loss.csv"), new_loss_df], ignore_index=True)
        else:
            loss_df = new_loss_df
        loss_df.to_csv("loss.csv", index=False)

    def save_model(self):
        """Save the model to a file"""

        # Make sure the models directory exists
        if not os.path.exists(os.path.join("data", "models")):
            os.mkdir(os.path.join("data", "models"))

        # Save the model
        torch.save(
            self.model.state_dict(),
            os.path.join("data", "models", f"{self.model_type}_{self.run_code}.pt")
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

    def load_model(self, run_code):
        """Load a trained model"""

        # Get configuration
        run_config_df = pd.read_csv("runs.csv")
        run_config = run_config_df[run_config_df["run_code"] == run_code].iloc[0]

        self.embedding_size = run_config.embedding_size.item()
        self.epochs = run_config.epochs.item()
        self.batch_size = run_config.batch_size.item()
        self.lr = run_config.lr.item()

        self.init_model(
            batch_size=self.batch_size, num_epochs=self.epochs, num_heads=run_config.num_heads.item(),
            dim_feedforward=run_config.dim_feedforward.item(), num_layers=run_config.num_layers.item(), lr=self.lr
        )

        print("Preparing the model for quantization ...")
        self.prepare_for_quantization()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Load the model state dictionary
        self.model.load_state_dict(torch.load(os.path.join("data", "models", f"{self.model_type}_{run_code}.pt")))


def add_to_batch(batch, vector):
    if batch is None:
        batch = vector.unsqueeze(0)
    else:
        batch = torch.cat([batch, vector.unsqueeze(0)])
    return batch


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
    print("Emptying train directory ...")
    for filename in tqdm(os.listdir(train_dir)):
        os.remove(os.path.join(train_dir, filename))
    print("Emptying test directory ...")
    for filename in tqdm(os.listdir(test_dir)):
        os.remove(os.path.join(test_dir, filename))

    # Pre-compute sequential chunks in each document for the PredictChunkDataset class
    sequential_chunks = {
        'video_id': [],
        'chunk_a': [],
        'chunk_b': [],
        'is_train': [],
    }

    # Iterate over each document, with tqdm to show progress
    print("Creating document chunks ...")
    for filename in tqdm(os.listdir(os.path.join("data", "youtube", "raw"))):
        # Load the document
        with open(os.path.join("data", "youtube", "raw", filename), "r") as f:
            doc = f.read()

        # Tokenize the document, and get the integer token ids
        tokenized_doc = word_tokenize(doc)
        doc_tokens = [EMBED_VOCAB.stoi[token] for token in tokenized_doc if token in EMBED_VOCAB.stoi]
        doc_words = [token for token in tokenized_doc if token in EMBED_VOCAB.stoi]

        # Generate random number to determine if this file is in the train or test set
        if random() < 0.8:
            chunk_dir = train_dir
        else:
            chunk_dir = test_dir

        # Chunk the document into CHUNK_SIZE-1 token chunks
        for i in range(0, len(doc_tokens), chunk_size):
            chunk_id = i // chunk_size

            # Convert tokens to embeddings
            embeddings = torch.tensor([embedding for embedding in doc_tokens[i:i+chunk_size]])
            # Pad the chunk with PAD_TOKEN_ID if it is smaller than CHUNK_SIZE
            embeddings = torch.nn.functional.pad(
                embeddings, (0, chunk_size - len(embeddings)), "constant", PAD_TOKEN_ID
            )

            # Get the video id from the filename
            video_id = filename[:filename.rfind(".")]

            # Save the chunk
            chunk_name = f"{video_id}_{chunk_id:04d}"
            if chunk_id > 0:
                sequential_chunks['video_id'].append(video_id)
                sequential_chunks['chunk_a'].append(chunk_id - 1)
                sequential_chunks['chunk_b'].append(chunk_id)
                sequential_chunks['is_train'].append(chunk_dir == train_dir)

            # Save the chunk
            torch.save(embeddings, os.path.join(chunk_dir, f"{chunk_name}.pt"))

            # Save the chunk as a text file
            with open(os.path.join(chunk_dir, f"{chunk_name}.txt"), "w") as f:
                f.write(" ".join(doc_words[i:i+chunk_size]))

    # Save the sequential chunks
    sequential_chunks_df = pd.DataFrame(sequential_chunks)
    sequential_chunks_df.to_csv(os.path.join("data", f"sequential-chunks-{chunk_size}.csv"), index=False)


def record_run(run_code, chunk_size, batch_size, epochs, embedding_size, num_heads, dim_feedforward, num_layers, lr):
    # Load data from previous runs
    if os.path.exists("runs.csv"):
        run_df = pd.read_csv("runs.csv")
    else:
        run_df = pd.DataFrame(columns=[
            "run_code", "chunk_size", "batch_size", "epochs", "embedding_size", "num_heads", "dim_feedforward",
            "num_layers", "lr"
        ])

    # Record the run hyperparameters for this run
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

    # Get the hyperparameters for this run
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
    parser.add_argument("--validate", action="store_true")

    # Add argument to choose either MLM or Dual model
    parser.add_argument("--model-type", type=str, default="dual")

    # Model hyper-parameters
    parser.add_argument("--chunk-size", type=int, default=64)

    # Model training hyperparameters
    train_group = parser.add_argument_group("model training")
    train_group.add_argument("--batch-size", type=int, default=8)
    train_group.add_argument("--dim-feedforward", type=int, default=512)
    train_group.add_argument("--epochs", type=int, default=None)
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
        # Check that the model type is valid
        if pargs.model_type not in ("mlm", "dual"):
            raise ValueError(f"Invalid model type: {pargs.model}. Must be 'mlm' or 'dual'.")

        trainer = DocumentEmbeddingTrainer(
            chunk_size=pargs.chunk_size, embedding_size=pargs.embedding_size, model_type=pargs.model_type
        )
        trainer.init_model(
            batch_size=pargs.batch_size, num_epochs=pargs.epochs, num_heads=pargs.num_heads,
            dim_feedforward=pargs.dim_feedforward, num_layers=pargs.num_layers, lr=pargs.lr
        )

        # Train the MLM model
        if pargs.model_type == "mlm":
            trainer.train_mlm()
        # Train the dual model
        elif pargs.model_type == "dual":
            trainer.train_dual()

        # Record the run results
        record_run(
            trainer.run_code, pargs.chunk_size, pargs.batch_size, pargs.epochs, pargs.embedding_size, pargs.num_heads,
            pargs.dim_feedforward, pargs.num_layers, pargs.lr
        )

    if pargs.validate:
        trainer = DocumentEmbeddingTrainer(run_code=pargs.run_code, model_type=pargs.model_type, is_test=True)
        trainer.load_model(pargs.run_code)

        my_r2, my_f1, my_accuracy = trainer.validate()
        print(f"r2: {my_r2}, f1: {my_f1}, accuracy: {my_accuracy}")
