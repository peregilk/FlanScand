import argparse
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

def get_embeddings(model, dataloader, emb_memmap, paths_memmap, total_size):
    with tqdm(total=total_size, desc="Processing records") as pbar:
        for data_batch, paths_batch, batch_indices in dataloader:
            embeddings = model.encode(data_batch, convert_to_tensor=True, show_progress_bar=False)
            for idx, global_idx in enumerate(batch_indices):
                emb_memmap[global_idx] = embeddings[idx].cpu().numpy()
                paths_memmap[global_idx] = paths_batch[idx]
            pbar.update(len(data_batch))

def jsonl_dataloader(file_path, batch_size=32):
    batch_data = []
    batch_paths = []
    batch_indices = []
    global_index = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            text = record['text']
            path = record['id']
            batch_data.append(text)
            batch_paths.append(path)
            batch_indices.append(global_index)
            global_index += 1
            if len(batch_data) == batch_size:
                yield batch_data, batch_paths, batch_indices
                batch_data, batch_paths, batch_indices = [], [], []
    if batch_data:
        yield batch_data, batch_paths, batch_indices

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    model = SentenceTransformer(args.model_name).to(device)

    dataset_size = sum(1 for line in open(args.input_file, 'r'))
    emb_size = args.emb_size  # Parameterized embedding size

    # Create directories if they don't exist
    os.makedirs(args.embeddings_dir, exist_ok=True)
    os.makedirs(args.paths_dir, exist_ok=True)

    # Parse the input file name to use it for output file names
    base_file_name = os.path.splitext(os.path.basename(args.input_file))[0]
    emb_file_path = os.path.join(args.embeddings_dir, f"{base_file_name}.memmap")
    path_file_path = os.path.join(args.paths_dir, f"{base_file_name}.memmap")

    emb_array = np.memmap(emb_file_path, dtype='float32', mode='w+', shape=(dataset_size, emb_size))
    path_array = np.memmap(path_file_path, dtype='U255', mode='w+', shape=(dataset_size,))

    loader = jsonl_dataloader(args.input_file, batch_size=args.batch_size)
    get_embeddings(model, loader, emb_array, path_array, dataset_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for text data in a jsonlines file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the jsonlines input file.")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings", help="Directory where embeddings memmap files will be stored.")
    parser.add_argument("--paths_dir", type=str, default="paths", help="Directory where paths memmap files will be stored.")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2", help="Model identifier for a pretrained Sentence Transformer model.")
    parser.add_argument("--emb_size", type=int, default=384, help="Dimension of the embeddings generated.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Number of records to process in each batch.")
    args = parser.parse_args()

    main(args)

