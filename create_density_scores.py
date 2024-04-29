import numpy as np
import argparse
import os
import pickle
from tqdm import tqdm
import logging

"""
Implements the data structure from the following papers.

1. Coleman, Benjamin, and Anshumali Shrivastava. "Sub-linear race sketches for approximate kernel density estimation on streaming data." Proceedings of The Web Conference 2020. 2020.
2. Coleman, Benjamin, Richard Baraniuk, and Anshumali Shrivastava. "Sub-linear memory sketches for near neighbor search on streaming data." International Conference on Machine Learning. PMLR, 2020.
3. Coleman, Benjamin, and Anshumali Shrivastava. "A one-pass distributed and private sketch for kernel sums with applications to machine learning at scale." Proceedings of the 2021 ACM SIGSAC Conference on Computer and Communications Security. 2021.
4. Coleman, Benjamin, et al. "One-pass diversified sampling with application to terabyte-scale genomic sequence streams." International Conference on Machine Learning. PMLR, 2022.
5. Liu, Zichang, et al. "One-Pass Distribution Sketch for Measuring Data Heterogeneity in Federated Learning." Advances in Neural Information Processing Systems 36 (2024).
"""

class RACE:
    def __init__(self, repetitions: int, hash_range: int, dtype=np.int32):
        self.dtype = dtype
        self.R = repetitions  # number of ACEs (rows) in the array
        self.W = hash_range  # range of each ACE (width of each row)
        self.counts = np.zeros((self.R, self.W), dtype=self.dtype)  # integer counters

    def add_batch(self, allhashes):
        allhashes = np.array(allhashes, dtype=int) % self.W
        for i in range(self.R):
            self.counts[i, :] += np.bincount(allhashes[i, :], minlength=self.W)

    def query_batch(self, allhashes):
        allhashes = np.array(allhashes, dtype=int) % self.W
        allhashes = allhashes.T
        values = np.zeros(allhashes.shape[0], dtype=float)
        N = np.sum(self.counts) / self.R
        for i, hashvalues in enumerate(allhashes):
            mean = 0
            for idx, hashvalue in enumerate(hashvalues):
                mean += self.counts[idx, hashvalue]
            values[i] = mean / (self.R * N)
        return values

class L2Hash:
    def __init__(self, N: int, d: int, r: float, seed: int = 0):
        self.d = d  # dimensionality of the data
        self.N = N  # number of hashes
        self.seed = seed  # random seed
        self.r = r  # bandwidth of L2 hash kernel
        self._init_projections()

    def _init_projections(self):
        np.random.seed(self.seed)
        self.W = np.random.normal(size=(self.N, self.d))
        self.b = np.random.uniform(low=0, high=self.r, size=self.N)

    def hash_batch(self, X):
        h = np.dot(self.W, X.T) + self.b[:, np.newaxis]
        h /= self.r
        return np.floor(h)

def print_stats(scores, num_bins=10):
    counts, bins = np.histogram(scores, bins=num_bins, range=(0, 1))
    total = counts.sum()
    print("\nProbability Distribution Bar Chart:")
    for count, bin_edge in zip(counts, bins):
        bin_width = 1 / num_bins
        print(f"{bin_edge:.2f} - {bin_edge + bin_width:.2f} | {'#' * int(50 * count / total)} ({count})")
    print("\nStatistics Table:")
    print("Range\t\t\tCount\tPercentage")
    for count, bin_edge in zip(counts, bins):
        bin_width = 1 / num_bins
        print(f"{bin_edge:.2f} - {bin_edge + bin_width:.2f}\t{count}\t{count / total * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate inverse propensity scores via KDE using locality sensitive hashing.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input memmap files")
    parser.add_argument("--output_folder", type=str, default="scores", help="Folder where memmap scores will be saved")
    parser.add_argument("--emb_size", type=int, default=384, help="Embedding size")
    parser.add_argument("--sketch_reps", type=int, default=1000, help="Number of hash functions (R): rows in sketch matrix")
    parser.add_argument("--sketch_range", type=int, default=20000, help="Width of sketch matrix (hash range B)")
    parser.add_argument("--kernel_bandwidth", type=float, default=0.1, help="Bandwidth of L2 hash kernel")
    parser.add_argument("--embedding_size", type=int, default=384, help="Size of the embeddings. 384 for MiniLM, 768 for BERT")
    parser.add_argument("--batch_size", type=int, default=16384, help="Number of embeddings to load at a time")
    parser.add_argument("--sketch_file", type=str, default=None, help="Path to load the sketch file, if it exists. Otherwise, we construct the sketch.")
    parser.add_argument("--nostats", action="store_true", help="Disable the printing of statistics")
    return parser.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    os.makedirs(args.output_folder, exist_ok=True)
    hash_fn = L2Hash(N=args.sketch_reps, d=args.embedding_size, r=args.kernel_bandwidth, seed=0)
    sketch = RACE(repetitions=args.sketch_reps, hash_range=args.sketch_range)

    # Load or construct the sketch
    if args.sketch_file and os.path.exists(args.sketch_file):
        with open(args.sketch_file, "rb") as f:
            sketch = pickle.load(f)
    else:
        logger.info("Constructing sketch from dataset")
        memmap_files = [f for f in os.listdir(args.input_folder) if f.endswith(".memmap")]
        if not memmap_files:
            logger.warning("No .memmap files found in the input directory.")
            return
        for filename in tqdm(memmap_files):
            file_path = os.path.join(args.input_folder, filename)
            dataset = np.memmap(file_path, dtype="float32", mode="r")
            num_embeddings = dataset.shape[0] // args.embedding_size
            dataset = dataset.reshape((num_embeddings, args.embedding_size))
            batch_nr = 0
            while True:
                offset_batch = args.batch_size * batch_nr
                dataset_batch = dataset[offset_batch:offset_batch + args.batch_size]
                if dataset_batch.shape[0] == 0:
                    break
                sketch.add_batch(hash_fn.hash_batch(dataset_batch))
                batch_nr += 1
        # Save the sketch object to a file
        sketch_file_path = os.path.join(args.output_folder, "sketch.pkl")
        with open(sketch_file_path, "wb") as f:
            pickle.dump(sketch, f)

    # Query the sketch for each embedding in the dataset to calculate sampling weights
    logger.info("Querying sketch for each embedding")
    for filename in tqdm(os.listdir(args.input_folder)):
        if filename.endswith(".memmap"):
            file_path = os.path.join(args.input_folder, filename)
            dataset = np.memmap(file_path, dtype="float32", mode="r")
            num_embeddings = dataset.shape[0] // args.embedding_size
            results = np.memmap(os.path.join(args.output_folder, filename.replace('.memmap', '_weights.memmap')),
                                dtype="float32", mode="w+", shape=(num_embeddings,))
            dataset = dataset.reshape((num_embeddings, args.embedding_size))
            batch_nr = 0
            while True:
                offset_batch = args.batch_size * batch_nr
                dataset_batch = dataset[offset_batch:offset_batch + args.batch_size]
                if dataset_batch.shape[0] == 0:
                    break
                scores = sketch.query_batch(hash_fn.hash_batch(dataset_batch))
                weights = 1 / (scores + 1e-8)  # Adding a small constant for numerical stability
                results[offset_batch:offset_batch + args.batch_size] = weights
                batch_nr += 1
            if not args.nostats:
                # Print normalised weights
                print_stats(weights/100)
                #print_stats(weights / np.max(weights))

if __name__ == "__main__":
    main()

