import argparse
import pandas as pd
import os

def convert_parquet_to_jsonl(input_dir, output_file):
    # List all Parquet files in the directory
    parquet_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.parquet')]

    # Load each Parquet file into a DataFrame and concatenate them
    df = pd.concat((pd.read_parquet(pf) for pf in parquet_files), ignore_index=True)

    # Write the combined DataFrame to a JSON Lines file
    df.to_json(output_file, orient='records', lines=True)

    print(f"Conversion complete. JSON Lines file created at: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert Parquet files to a single JSON Lines file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing Parquet files")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON Lines file path")
    args = parser.parse_args()

    convert_parquet_to_jsonl(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()

