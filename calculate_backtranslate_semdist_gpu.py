import argparse
import jsonlines
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

def calculate_semantic_distance(input_file, output_file, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', batch_size=32):
    model = SentenceTransformer(model_name)

    with jsonlines.open(input_file, mode='r') as reader, jsonlines.open(output_file, mode='w') as writer:
        batch = []
        for obj in tqdm(reader, desc="Processing records"):
            batch.append(obj)
            if len(batch) >= batch_size:
                process_batch(batch, writer, model)
                batch = []
        
        if batch:
            process_batch(batch, writer, model)

def process_batch(batch, writer, model):
    # Precompute embeddings for both sets of fields to avoid recomputation and utilize batch processing efficiently
    inputs_texts = [obj['inputs'] for obj in batch if 'inputs' in obj and 'inputs_backtranslation' in obj]
    inputs_back_texts = [obj['inputs_backtranslation'] for obj in batch if 'inputs' in obj and 'inputs_backtranslation' in obj]
    targets_texts = [obj['targets'] for obj in batch if 'targets' in obj and 'targets_backtranslation' in obj]
    targets_back_texts = [obj['targets_backtranslation'] for obj in batch if 'targets' in obj and 'targets_backtranslation' in obj]

    # Encode in batches
    if inputs_texts:
        inputs_embeddings = model.encode(inputs_texts, convert_to_tensor=True)
        inputs_back_embeddings = model.encode(inputs_back_texts, convert_to_tensor=True)
        inputs_cosine_sims = util.pytorch_cos_sim(inputs_embeddings, inputs_back_embeddings)

    if targets_texts:
        targets_embeddings = model.encode(targets_texts, convert_to_tensor=True)
        targets_back_embeddings = model.encode(targets_back_texts, convert_to_tensor=True)
        targets_cosine_sims = util.pytorch_cos_sim(targets_embeddings, targets_back_embeddings)

    for i, obj in enumerate(batch):
        if 'inputs' in obj and 'inputs_backtranslation' in obj:
            semantic_distance = 1 - inputs_cosine_sims[i][i].item()
            obj['inputs_semdist'] = semantic_distance
        if 'targets' in obj and 'targets_backtranslation' in obj:
            semantic_distance = 1 - targets_cosine_sims[i][i].item()
            obj['targets_semdist'] = semantic_distance
        writer.write(obj)

def main():
    parser = argparse.ArgumentParser(description='Calculate semantic distances in a JSON Lines file.')
    parser.add_argument('--input_file', type=str, required=True, help='Input JSON Lines file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON Lines file path')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')

    args = parser.parse_args()
    
    calculate_semantic_distance(args.input_file, args.output_file, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
