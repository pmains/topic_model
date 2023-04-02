#!/bin/bash

for chunk_size in 64 128 256; do
  for num_layers in 1 2 4 6; do
    for embedding_size in 64 128 256; do
      for batch_size in 8 16 32; do
        for lr in 1e-3 1e-4 1e-5; do
          python doc_embed_torch.py --train --chunk-size $chunk_size --batch-size $batch_size --num-layers $num_layers --embedding-size $embedding_size --lr $lr || exit 1
        done
      done
    done
  done
done
