import numpy as np
import umap
import matplotlib.pyplot as plt
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize embeddings from .memmap files using UMAP.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input .memmap files")
    parser.add_argument("--output_file", type=str, default="embedding_visualization.png", help="Output image file")
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of embeddings to sample for visualization")
    parser.add_argument("--embedding_size", type=int, default=384, help="Size of the embeddings")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load embeddings from .memmap files
    embeddings = []
    for filename in os.listdir(args.input_folder):
        if filename.endswith(".memmap"):
            file_path = os.path.join(args.input_folder, filename)
            dataset = np.memmap(file_path, dtype="float32", mode="r")
            num_embeddings = dataset.shape[0] // args.embedding_size
            dataset = dataset.reshape((num_embeddings, args.embedding_size))
            embeddings.append(dataset)

    embeddings = np.concatenate(embeddings, axis=0)

    # Sample a subset of embeddings for visualization
    num_samples = min(args.num_samples, embeddings.shape[0])
    sampled_indices = np.random.choice(embeddings.shape[0], num_samples, replace=False)
    sampled_embeddings = embeddings[sampled_indices]

    # Apply UMAP for dimensionality reduction
    umap_embeddings = umap.UMAP(n_components=2, random_state=42).fit_transform(sampled_embeddings)

    # Visualize the embeddings using a scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5, alpha=0.5)
    plt.title("Embedding Visualization (UMAP)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(args.output_file)
    plt.close()

if __name__ == "__main__":
    main()
