import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def sim(a, b):
    """Calculate the cosine similarity between two vectors"""

    # Compute dot product and norms
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Compute cosine similarity
    return dot_product / (norm_a * norm_b)


def get_weighted_mean(te, ti_counts):
    """Get the weighted mean of the topic embeddings"""

    # Get the weights of each topic
    ti_weights = np.array([x / sum(ti_counts) for x in ti_counts])
    # Multiply the topic embeddings by the weights
    te_weighted = np.sum(te * ti_weights[:, np.newaxis], axis=0)

    return te_weighted


def compare_models(era1, era2, cat, cat2=None, run_code=None):
    if cat2 is None:
        cat2 = cat

    # Use the run code to load the correct topic embeddings and info
    if run_code is not None:
        te1_file = f"data/topics/{era1}_{cat}_topic_embeddings_{run_code}.npy"
        te2_file = f"data/topics/{era2}_{cat2}_topic_embeddings_{run_code}.npy"
        ti1_file = f"data/topics/{era1}_{cat}_topic_info_{run_code}.csv"
        ti2_file = f"data/topics/{era2}_{cat2}_topic_info_{run_code}.csv"
    # Or use the BERT embeddings
    else:
        te1_file = f"data/topics/{era1}_{cat}_topic_embeddings.npy"
        te2_file = f"data/topics/{era2}_{cat2}_topic_embeddings.npy"
        ti1_file = f"data/topics/{era1}_{cat}_topic_info.csv"
        ti2_file = f"data/topics/{era2}_{cat2}_topic_info.csv"

    for file_name in [te1_file, te2_file, ti1_file, ti2_file]:
        if not os.path.exists(file_name):
            return None

    # Load the topic embeddings and info
    te1 = np.load(te1_file, allow_pickle=True)
    te2 = np.load(te2_file, allow_pickle=True)

    if len(te1.shape) == 0 or len(te2.shape) == 0:
        print(f"Unable to compare {era1}_{cat} and {era2}_{cat2}")
        return None

    ti1 = pd.read_csv(ti1_file)
    ti2 = pd.read_csv(ti2_file)

    # Get weighted means of the topic embeddings including the outlier topic
    ti1_counts = ti1.Count.values.tolist()
    ti2_counts = ti2.Count.values.tolist()
    # Multiple the topic embeddings by the weights
    te1_weighted = get_weighted_mean(te1, ti1_counts)
    te2_weighted = get_weighted_mean(te2, ti2_counts)

    # Calculate the cosine similarity between the two weighted embeddings
    weighted_cosine_sim = sim(te1_weighted, te2_weighted)

    # Get weighted means of the topic embeddings excluding the outlier topic
    ti1_counts_no_outliers = ti1_counts[1:]
    ti2_counts_no_outliers = ti2_counts[1:]
    # Multiple the topic embeddings by the weights
    te1_weighted_no_outliers = get_weighted_mean(te1[1:], ti1_counts_no_outliers)
    te2_weighted_no_outliers = get_weighted_mean(te2[1:], ti2_counts_no_outliers)

    # Calculate the cosine similarity between the two weighted embeddings
    weighted_cosine_sim_no_outliers = sim(te1_weighted_no_outliers, te2_weighted_no_outliers)

    # Calculate the cosine similarity between the two embeddings
    cosine_sim = cosine_similarity(te1, te2)
    # Continuity is the average similarity of the most similar topic
    continuity = np.mean(np.max(cosine_sim, axis=1))
    # Adjusted continuity normalizes the similarity by the number of topics
    adjusted_continuity = continuity / len(te1)

    # Stability

    # Calculate the mean cosine similarity
    mean_similarity = sim(np.mean(te1, axis=0), np.mean(te2, axis=0))
    # Calculate the same, but exclude -1, the outlier topic
    mean_similarity_no_outliers = sim(np.mean(te1[1:], axis=0), np.mean(te2[1:], axis=0))

    return {
        'era1': era1,
        'era2': era2,
        'cat1': cat,
        'cat2': cat2,
        'weighted_sim': weighted_cosine_sim,
        'weighted_sim_no_outliers': weighted_cosine_sim_no_outliers,
        'mean_similarity': mean_similarity,
        'mean_similarity_no_outliers': mean_similarity_no_outliers,
        'continuity': continuity,
        'adjusted_continuity': adjusted_continuity,
    }


if __name__ == "__main__":
    # Accept --run-code argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-code', type=str, default=None)
    args = parser.parse_args()

    df = pd.DataFrame(columns=[
        'era1', 'era2', 'cat1', 'cat2', 'weighted_sim', 'weighted_sim_no_outliers', 'mean_similarity',
        'mean_similarity_no_outliers', 'continuity', 'adjusted_continuity'
    ])

    house_eras = (
        'JB-H2', 'JB-H1', 'DT-H4', 'JB-H3', 'DT-H3', 'DT-H2', 'DT-H1', 'BO-H8', 'BO-H7', 'BO-H6', 'BO-H5', 'BO-H4',
        'BO-H3', 'BO-H2', 'BO-H1'
    )

    categories = ('democrat', 'republican', 'all')

    # Compare all combinations of eras and categories
    for my_era1 in house_eras:
        for my_era2 in house_eras:
            for my_cat1 in categories:
                for my_cat2 in categories:
                    # Don't compare the same era and category
                    if my_era1 == my_era2 and my_cat1 == my_cat2:
                        continue

                    compare_data = compare_models(my_era1, my_era2, my_cat1, my_cat2, args.run_code)
                    if compare_data is not None:
                        print(f"Comparing {my_era1}_{my_cat1} and {my_era2}_{my_cat2}")
                        df = pd.concat([df, pd.DataFrame(compare_data, index=[0])], ignore_index=True)

    # Save the results
    if args.run_code is not None:
        df.to_csv(f'data/compare_models_{args.run_code}.csv', index=False)
    else:
        df.to_csv('data/compare_models.csv', index=False)
