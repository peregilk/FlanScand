import argparse
import json
import os
from pathlib import Path
import groq
from tqdm import tqdm

def process_file(input_file, output_file, dryrun, model, temperature, language):
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key is None:
        print("GROQ_API_KEY environment variable is not set.")
        return

    client = groq.Client(api_key=api_key)
    valid_outputs = ["1", "2", "3", "4", "5"]
    language_map = {
        "nor": "Norwegian",
        "non": "Nynorsk",
        "swe": "Swedish",
        "dan": "Danish"
    }
    lang_name = language_map.get(language, "Norwegian")  # Default to Norwegian if invalid code

    system_message = (
        f"As an experienced translator proficient in both {lang_name} and English, your task is to critically evaluate a "
        f"{lang_name} translation originating from an English text. This translation is crucial to an instruct tuning dataset "
        "that demands the highest levels of accuracy and natural language fluidity. Your analysis must rigorously assess "
        "the logical consistency across both languages, ensuring absolute alignment in answers and results. "
        "\n\nPlease note: The text strictly adheres to a format marked with #ENGLISH#, #NORWEGIAN#, #INPUTS#, #OUTPUTS#, and #END#. "
        "Within this framework, your task is to methodically evaluate the Norwegian translation and then try to improve it. "
        "Note that your translation should be split into the input part and the output part. After thorough review, "
        "you are required to rate the translation quality on a scale from 1 (poor) to 5 (perfect). You must only produce a numerical "
        "rating as your output. Ensure that your rating strictly consists of one digit: 1, 2, 3, 4, or 5. "
        "\n\nPresent your output as a valid JSON object. This object should have three fields: a 'rating' field containing "
        "the numerical value (1, 2, 3, 4, or 5), an 'improved_translation_inputs' field containing your suggestion for a better "
        f"{lang_name} translation of the English input, and an 'improved_translation_targets' field containing your suggestion for a better {lang_name} translation of the English target. "
        "Make sure characters are escaped if needed."
    )

    total_prompt_tokens = 0
    total_completion_tokens = 0

    with open(input_file, "r") as f, open(output_file, "w") as writer:
        pbar = tqdm(desc="Processing", total=None)
        for line in f:
            obj = json.loads(line)
            pbar.update(1)
            template = (
                "#ENGLISH#\n"
                "#INPUTS#\n"
                f"{obj['inputs']}\n"
                "#TARGETS#\n"
                f"{obj['targets']}\n"
                "#END#\n\n"
                f"#{lang_name.upper()}#\n"
                "#INPUTS#\n"
                f"{obj[f'inputs_{language}']}\n"
                "#OUTPUTS#\n"
                f"{obj[f'targets_{language}']}\n"
                "#END#"
            )

            if dryrun:
                print(f"Processing Index {obj['index']}: Dry-run mode")
                print(f"System Message: {system_message}")
                print(f"API Payload:\n{template}\n")
            else:
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": template}
                        ],
                        temperature=temperature,
                        max_tokens=2048,
                        top_p=1,
                        stream=False,
                        seed=1234,
                        stop=None,
                        response_format={"type": "json_object"}
                    )
                    usage = completion.usage
                    total_prompt_tokens += usage.prompt_tokens
                    total_completion_tokens += usage.completion_tokens

                    
                    content = completion.choices[0].message.content
                    if content:
                        try:
                            json_response = json.loads(content)
                            rating = json_response.get("rating")
                            improved_input = json_response.get("improved_translation_inputs")
                            improved_output = json_response.get("improved_translation_targets")
                            
                            obj[f"{model}_rating"] = rating if str(rating) in valid_outputs else "Invalid"
                            obj[f"{model}_improved_inputs"] = improved_input
                            obj[f"{model}_improved_targets"] = improved_output
                        except json.JSONDecodeError:
                            print(f"Invalid JSON response on {obj['index']}: {content}")
                            obj[f"{model}_rating"] = "Invalid"
                    else:
                        print(f"No output on {obj['index']}")
                        obj[f"{model}_rating"] = "Invalid"

                    writer.write(json.dumps(obj, ensure_ascii=False) + "\n")

                except groq.BadRequestError as e:
                    print(f"API Error on {obj['index']}: {str(e)}")
                    obj[f"{model}_rating"] = "Invalid"
                
        pbar.close()

        print(f"Total Prompt Tokens: {total_prompt_tokens}")
        print(f"Total Completion Tokens: {total_completion_tokens}")
        print(f"Estimated Cost: ${(total_prompt_tokens * 0.05 + total_completion_tokens * 0.10) / 1_000_000:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Process a JSON-lines file for language evaluation using the Groq API.")
    parser.add_argument("--input_file", type=str, required=True, help="The input JSON-lines file path")
    parser.add_argument("--output_file", type=str, required=True, help="The output JSON-lines file path")
    parser.add_argument("--model", type=str, default="llama3-8b-8192", choices=["llama3-8b-8192", "llama3-70b-8192"], help="Model name for the API call")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature setting for the API call")
    parser.add_argument("--language", type=str, default="nor", choices=["nor", "non", "swe", "dan"], help="Language code for the translation evaluation")
    parser.add_argument("--dryrun", action="store_true", help="Run in dry-run mode to print API call data without executing")
    args = parser.parse_args()

    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    process_file(args.input_file, args.output_file, args.dryrun, args.model, args.temperature, args.language)

if __name__ == "__main__":
    main()
