import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from doc_embed_torch import DocumentEmbeddingTrainer, record_run

# Define the search space
space = [
    Real(1e-6, 1e-1, name="lr", prior="log-uniform"),
    Integer(0, 2, name="chunk_size"),
    Integer(0, 3, name="num_layers"),
    Integer(0, 5, name="batch_size"),
    Integer(0, 4, name="embedding_size"),
    Integer(0, 3, name="num_heads"),
    Integer(0, 4, name="dim_feedforward"),
]


# Define the objective function to minimize
@use_named_args(space)
def objective(lr, chunk_size, num_layers, batch_size, embedding_size, num_heads, dim_feedforward):
    chunk_size = [64, 128, 256][chunk_size]  # chunk_size is 64, 128 or 256
    dim_feedforward = [64, 128, 256, 512, 1024][dim_feedforward]  # dim_feedforward is 64, 128, 256, 512 or 1024
    embedding_size = [64, 128, 256, 512, 1024][embedding_size]  # embedding_size is 64, 128, 256, 512 or 1024
    num_heads = [1, 2, 4, 8][num_heads]  # num_heads is 1, 2, 4 or 8
    num_layers = [1, 2, 4, 6][num_layers]  # num_layers is 1, 2, 4 or 6
    batch_size = [1, 2, 4, 8, 16, 32][batch_size]  # batch_size is 1, 2, 4, 8, 16 or 32

    # Create a new instance of the trainer with the specified hyper-parameters
    trainer = DocumentEmbeddingTrainer(chunk_size=chunk_size, embedding_size=embedding_size)
    epochs = 100
    trainer.init_mlm(
        batch_size=batch_size,
        num_epochs=epochs,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers,
        lr=lr
    )
    print(
        f"Training {trainer.run_code} with lr={lr}, chunk_size={chunk_size}, num_layers={num_layers}, "
        f"batch_size={batch_size}, embedding_size={embedding_size}, num_heads={num_heads},"
        f"dim_feedforward={dim_feedforward} for {epochs} epochs"
    )

    # Train the model
    trainer.train()

    # Read the loss from the file and return the minimum loss
    loss_df = pd.read_csv("loss.csv")
    # Ignore different runs
    loss_df = loss_df[loss_df["run_code"] == trainer.run_code]

    record_run(
        trainer.run_code, chunk_size, batch_size, epochs, embedding_size, num_heads, dim_feedforward, num_layers, lr
    )
    
    # Return the minimum loss
    return np.min(loss_df["loss"].values)


# Perform the Bayesian optimization search
res_gp = gp_minimize(
    objective, space, n_calls=50, n_random_starts=10, random_state=42
)

# Print the results
print(f"Best loss: {res_gp.fun:.4f}")
print(f"Best hyper-parameters: {res_gp.x}")
