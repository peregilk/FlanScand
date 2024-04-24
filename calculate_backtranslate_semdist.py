import argparse
import jsonlines
from sentence_transformers import SentenceTransformer, util

def calculate_semantic_distance(input_file, output_file, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
    # Load the model
    model = SentenceTransformer(model_name)
    
    # Open the input and output jsonlines files
    with jsonlines.open(input_file, mode='r') as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in reader:
            # Handle inputs and inputs_backtranslation
            if 'inputs' in obj and 'inputs_backtranslation' in obj:
                text1 = obj['inputs']
                text2 = obj['inputs_backtranslation']
                embeddings1 = model.encode(text1, convert_to_tensor=True)
                embeddings2 = model.encode(text2, convert_to_tensor=True)
                cosine_sim = util.pytorch_cos_sim(embeddings1, embeddings2)
                semantic_distance = 1 - cosine_sim.item()
                obj['inputs_semdist'] = semantic_distance

            # Handle targets and targets_backtranslation
            if 'targets' in obj and 'targets_backtranslation' in obj:
                text1 = obj['targets']
                text2 = obj['targets_backtranslation']
                embeddings1 = model.encode(text1, convert_to_tensor=True)
                embeddings2 = model.encode(text2, convert_to_tensor=True)
                cosine_sim = util.pytorch_cos_sim(embeddings1, embeddings2)
                semantic_distance = 1 - cosine_sim.item()
                obj['targets_semdist'] = semantic_distance
            
            # Write the modified object to the output file
            writer.write(obj)

def main():
    parser = argparse.ArgumentParser(description='Calculate semantic distances in a JSON Lines file.')
    parser.add_argument('--input_file', type=str, required=True, help='Input JSON Lines file path')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON Lines file path')
    
    args = parser.parse_args()
    
    calculate_semantic_distance(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
