import os
import re

import dotenv
import openai
from nltk import word_tokenize

from wwp_gpt import issue_command


def clean_text_gpt(text):
    command_text = (
        "Format the following text as a transcript. The text needs proper capitalization, punctuation, and paragraph "
        "breaks. Even though the text might start mid-sentence, print the entirety of the text as provided without"
        "any outside content or context."
    )

    print('Cleaning text...')
    complete = False
    text_fragments = [text]
    while not complete:
        try:
            print(f"Cleaning {len(text_fragments)} fragments...")
            print(text_fragments)
            # Process all text fragments and concatenate the results
            result_text = ' '.join([issue_command(command_text, fragment) for fragment in text_fragments])
            # Remove extra space and newlines between paragraphs
            result_text = re.sub(r'\s*\n[\s\n]+', '\n', result_text)
            complete = True
        except openai.error.InvalidRequestError:
            print("Error, splitting text...")
            # Tokenize the fragments and split them in half
            new_text_fragments = []
            for fragment in text_fragments:
                fragment_tokens = word_tokenize(fragment)
                fragment_length = len(fragment_tokens)
                # If the fragment is too short, skip it
                if fragment_length < 10:
                    continue
                # Split the fragment in half
                split_index = fragment_length // 2
                new_text_fragments.append(' '.join(fragment_tokens[:split_index]))
                new_text_fragments.append(' '.join(fragment_tokens[split_index:]))

            # Overwrite the original fragments
            text_fragments = new_text_fragments

    return result_text


def load_api():
    # Load the API key from the .env file
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


def main():
    load_api()

    # Set up the paths
    raw_dir_path = os.path.join('data', 'youtube', 'raw')
    clean_dir_path = os.path.join('data', 'youtube', 'clean')

    file_names = os.listdir(raw_dir_path)

    for file_name in file_names:
        # If clean file already exists, skip
        if os.path.exists(os.path.join(clean_dir_path, file_name)):
            continue

        # Read in the file
        with open(os.path.join(raw_dir_path, file_name), 'r') as f:
            raw_text = f.read()
            print(f'Cleaning {file_name}...')
            # Clean the text
            try:
                # Clean the text
                cleaned_text = clean_text_gpt(raw_text)
                # Remove extra newlines from multiple paragraphs
                cleaned_text = re.sub(r'\n[\s\n]+', '\n', cleaned_text)
            except ValueError:
                # If there is an error, skip the file
                print(f'Error with {file_name}')
                continue

            # Write the cleaned text to a file
            with open(os.path.join(clean_dir_path, file_name), 'w') as clean_file:
                clean_file.write(cleaned_text)

        # Break after the first video
        break


def long_text_test():
    load_api()

    file_name = os.path.join("data", "youtube", "raw", "u.s.representativeelissasl7729-i7s_AhXIWUk.txt")
    with open(file_name, 'r') as file_handle:
        raw_text = file_handle.read()
    clean_text = clean_text_gpt(raw_text)
    print("Clean text: ", clean_text)


if __name__ == '__main__':
    main()
    # long_text_test()
