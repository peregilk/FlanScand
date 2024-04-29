import numpy as np
import argparse
import os

def normalize_embeddings(embeddings):
    """ Normalize the embeddings to unit norm (l2-norm). """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings

def process_memmap_files(input_dir, output_dir, emb_size, local):
    """ Process all memmap files in the specified directory, normalize and save to a new directory. """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist

    if not local:
        # Global normalization across all files
        all_embeddings = []

        # Load embeddings from all files
        for filename in os.listdir(input_dir):
            if filename.endswith('.memmap'):
                file_path = os.path.join(input_dir, filename)
                embeddings = np.memmap(file_path, dtype='float32', mode='r')
                num_embeddings = embeddings.shape[0] // emb_size
                embeddings = embeddings[:num_embeddings * emb_size].reshape((num_embeddings, emb_size))
                all_embeddings.append(embeddings)
                print(f"Input shape for {filename}: {embeddings.shape}")

        # Concatenate embeddings from all files
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Normalize the embeddings globally
        normalized_embeddings = normalize_embeddings(all_embeddings)

        # Split the normalized embeddings back into individual files
        start_index = 0
        for filename in os.listdir(input_dir):
            if filename.endswith('.memmap'):
                file_path = os.path.join(input_dir, filename)
                embeddings = np.memmap(file_path, dtype='float32', mode='r')
                num_embeddings = embeddings.shape[0] // emb_size
                end_index = start_index + num_embeddings
                normalized_file_embeddings = normalized_embeddings[start_index:end_index]
                output_file_path = os.path.join(output_dir, filename)
                output_memmap = np.memmap(output_file_path, dtype='float32', mode='w+', shape=(num_embeddings, emb_size))
                output_memmap[:] = normalized_file_embeddings
                print(f"Processed and saved normalized embeddings for {filename}. Shape: {output_memmap.shape}")
                start_index = end_index
    else:
        # Local normalization within each file
        for filename in os.listdir(input_dir):
            if filename.endswith('.memmap'):
                file_path = os.path.join(input_dir, filename)
                output_file_path = os.path.join(output_dir, filename)

                # Load embeddings as a memmap file
                embeddings = np.memmap(file_path, dtype='float32', mode='r')
                print(f"Input shape for {filename}: {embeddings.shape}")

                # Reshape the loaded embeddings
                num_embeddings = embeddings.shape[0] // emb_size
                embeddings_2d = embeddings[:num_embeddings * emb_size].reshape((num_embeddings, emb_size))

                # Normalize the embeddings
                normalized_embeddings = normalize_embeddings(embeddings_2d)

                # Save the normalized embeddings as a new memmap file with the same shape as input
                output_memmap = np.memmap(output_file_path, dtype='float32', mode='w+', shape=(num_embeddings, emb_size))
                output_memmap[:] = normalized_embeddings
                print(f"Processed and saved normalized embeddings for {filename}. Shape: {output_memmap.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize embeddings stored in numpy memmap format.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input memmap files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder where normalized memmap files will be saved")
    parser.add_argument("--emb_size", type=int, default=384, help="Embedding size")
    parser.add_argument("--local", action='store_true', help="Normalize embeddings locally within each file")

    args = parser.parse_args()

    # Process all memmap files in the specified directories
    process_memmap_files(args.input_folder, args.output_folder, args.emb_size, args.local)
