from datasets import load_dataset
import pandas as pd
import nltk
import contractions
import re
from sentence_transformers import SentenceTransformer, util
import torch

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# NLTK downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def is_complete_sentence(sentence):
    tokens = word_tokenize(sentence)
    tags = pos_tag(tokens)
    has_subject = False
    has_verb = False
    for word, tag in tags:
        if tag.startswith('NN') or tag.startswith('PRP'):  
            has_subject = True
        if tag.startswith('VB'):  
            has_verb = True
    return has_subject and has_verb

def get_text_until_first_verb(sentence):
    tokens = word_tokenize(sentence)
    tags = pos_tag(tokens)
    for i, (word, tag) in enumerate(tags):
        if tag.startswith('VB'):  
            return ' '.join(tokens[:i + 1])
    return None

def expand_contractions_in_sentence(sentence):
    return contractions.fix(sentence)

def has_single_quote(sentence):
    count_single_quote = sentence.count("'")
    count_double_quote = sentence.count('"')
    count_backtick = sentence.count('`')
    total_quotes = count_single_quote + count_double_quote + count_backtick
    return total_quotes % 2 != 0

def starts_with_double_backtick(sentence):
    return sentence.startswith('``')

def normalize_sentence(sentence):
    sentence = sentence.strip()
    sentence = expand_contractions_in_sentence(sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def is_similar(new_embedding, existing_embeddings, threshold=0.8):
    if len(existing_embeddings) == 0:
        return False
    # Stack existing embeddings into a tensor
    existing_embeddings_tensor = torch.stack(existing_embeddings)  # Shape: (N, embedding_dim)
    # Compute cosine similarities
    similarities = util.cos_sim(new_embedding, existing_embeddings_tensor)  # Shape: (1, N)
    max_similarity = similarities.max().item()  # Extract scalar value
    return max_similarity >= threshold

def ends_with_verb(sentence):
    tokens = word_tokenize(sentence)
    if not tokens:
        return False  # Return False if the sentence is empty
    last_word = tokens[-1]
    # Get the part-of-speech tag of the last word
    last_word_tag = pos_tag([last_word])[0][1]
    # Check if the part-of-speech tag starts with a verb (VB, VBD, VBG, VBN, VBP, VBZ)
    return last_word_tag.startswith('VB')

def process_dataset(sample_size=10000, limit=5000):
    #Embedding Loaded
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Data loading
    dataset = load_dataset('bookcorpus')

    x_i_list = []
    x_i_set = set()
    existing_embeddings = []


    for sample in dataset['train']:
        text = sample['text']
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # Exclude sentences starting with "`"
            if starts_with_double_backtick(sentence):
                continue
            # Sentence length check
            if len(sentence.split()) < 5 or len(sentence.split()) > 30:
                continue
            # Sentence completeness check
            if is_complete_sentence(sentence):
                x_i = get_text_until_first_verb(sentence)
                # Check if x_i is not None and satisfies the minimum word count condition
                if x_i and len(x_i.split()) >= 3:
                    # Expand contractions
                    x_i = expand_contractions_in_sentence(x_i)
                    # Exclude sentences with a single quote
                    if has_single_quote(x_i):
                        continue
                    # Normalize sentence
                    x_i_normalized = normalize_sentence(x_i)
                    # Check for duplicates
                    if x_i_normalized in x_i_set:
                        continue
                    # Compute new embedding
                    new_embedding = model.encode(x_i, convert_to_tensor=True)
                    # Check for similarity
                    if is_similar(new_embedding, existing_embeddings, threshold=0.8):
                        continue  # Similar sentence exists, skip
                    # Add to datasets
                    x_i_set.add(x_i_normalized)
                    x_i_list.append(x_i)
                    existing_embeddings.append(new_embedding)
            # Break if desired sample size reached
            if len(x_i_list) >= 10000:
                break
        if len(x_i_list) >= 10000:
            break
        
        # Check if each sentence in the 'X_i' column ends with a verb and store in a new column
    df = pd.DataFrame({'X_i': x_i_list})
    df['ends_with_verb'] = df['X_i'].apply(ends_with_verb)

    # Select only sentences ending with a verb
    filtered_df = df[df['ends_with_verb'] == True]

    # Delete unnecessary columns
    filtered_df = filtered_df.drop(columns=['ends_with_verb'])
    filtered_df = filtered_df[0:5000]

    return filtered_df
# Save to CSV
# df = pd.DataFrame({'X_i': x_i_list})
# df.to_csv('X_i_samples_20240923.csv', index=False)


def save_to_csv(df, filename='X_i_samples_5k.csv'):
    df.to_csv(filename, index=False)
    
# Main function to be called from main.ipynb
def generate_dataset(sample_size=10000, limit=5000, output_file='X_i_samples_5k.csv'):
    df = process_dataset(sample_size, limit)
    save_to_csv(df, output_file)
    return df
