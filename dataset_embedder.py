import csv
from openai import OpenAI
from os import getenv

# Set up OpenAI API key
api_key = "sk-None-h43oaqtxSzuBlil0BsqnT3BlbkFJWyaGnIEa2bH3oFncqvFk"
client = OpenAI(api_key = api_key)


def generate_embedding(text):
    try:
        response = client.embeddings.create(input = text, model = "text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for text '{text}': {e}")
        return None


def process_csv(input_file, output_file):
    with open(input_file, 'r', newline = '', encoding = 'utf-8') as infile, \
            open(output_file, 'w', newline = '', encoding = 'utf-8') as outfile:
        reader = csv.DictReader(infile)

        # Check for the correct column names
        if 'Question' not in reader.fieldnames or 'Answer' not in reader.fieldnames:
            raise ValueError("Input CSV must contain 'Question' and 'Answer' columns")

        fieldnames = reader.fieldnames + ['text', 'embedding']
        writer = csv.DictWriter(outfile, fieldnames = fieldnames)

        writer.writeheader()

        for row in reader:
            question = row['Question']
            answer = row['Answer']
            text = f"{question} {answer}"
            embedding = generate_embedding(text)

            new_row = row.copy()  # Keep all original columns
            new_row['text'] = text
            new_row['embedding'] = embedding
            writer.writerow(new_row)


if __name__ == "__main__":
    # input_file = "../kakao_fastapi/biztalk_dataset_sheets.csv"
    # output_file = "../kakao_fastapi/biztalk_output.csv"
    input_file = "dataset.csv"
    output_file = "dataset_output.csv"
    process_csv(input_file, output_file)
    print(f"Processing complete. Output written to {output_file}")


