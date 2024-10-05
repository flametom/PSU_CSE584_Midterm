import anthropic
import pandas as pd
from tqdm import tqdm
import re
import logging
import time
from difflib import SequenceMatcher

def clean_text(text, x_i):
    # Replace newline characters and multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    x_i = re.sub(r'\s+', ' ', x_i).strip()
    
    # Compare the beginning of x_i and text
    matcher = SequenceMatcher(None, x_i.lower(), text.lower())
    match = matcher.find_longest_match(0, len(x_i), 0, len(text))
    
    if match.size > 0:
        # If part of x_i matches the beginning of text
        text_start = text[:match.b + match.size]
        text_end = text[match.b + match.size:].lstrip()
        
        # Update text_start while maintaining the original format of x_i
        updated_start = x_i[:match.a + match.size]
        
        text = updated_start + " " + text_end
    else:
        # If there's no match, use x_i as is
        text = x_i + " " + text
    
    # Define sentence end pattern (period, exclamation mark, or question mark followed by a space or end of string)
    sentence_end_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    
    # Split into sentences
    sentences = re.split(sentence_end_pattern, text)
    
    if not sentences:
        return "Invalid generation"
    
    # Select the first sentence
    sentence = sentences[0].strip()
    
    # Add a period if the sentence is incomplete (no punctuation at the end)
    if not sentence.endswith(('.', '!', '?')):
        sentence += '.'
    
    # Remove duplicate words (case-insensitive)
    words = sentence.split()
    unique_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() != words[i-1].lower():
            unique_words.append(word)
    
    final_text = ' '.join(unique_words)
    
    return final_text

def is_valid_sentence(text, x_i):
    # Check for empty string
    if not text or text == "Invalid generation":
        return False
    
    # Check if it starts with X_i (case-insensitive, ignoring spaces)
    if not text.lower().replace(" ", "").startswith(x_i.lower().replace(" ", "")):
        return False
    
    # Limit to minimum 3 words added, maximum 50 words
    words = text.split()
    x_i_words = x_i.split()
    if len(words) < 3:
        return False
    if len(words) > 50:
        return False
    
    # Check if the sentence contains an underscore ('_')
    if '_' in text:
        return False
    
    # Check for unnecessary starting phrases
    if re.match(r'^(Text:|Solution:|Answer:|B:)\s*', text, flags=re.IGNORECASE):
        return False
    
    # Check if the sentence ends with a period, question mark, or exclamation mark
    if not text.endswith(('.', '!', '?')):
        return False
    
    # Check for matching quotation marks
    if text.count('"') % 2 != 0:
        return False
    
    # Check for repetition
    if has_repetition(words):
        return False
    
    return True

def has_repetition(words):
    # Check for repetition of 3 or more consecutive words
    for i in range(len(words) - 2):
        if words[i:i+3] == words[i+3:i+6]:
            return True
    return False

def generate_text_with_claude(prompt, client):
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            temperature=0.7,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        generated_text = response.content[0].text
        return generated_text
    
    except Exception as e:
        logging.error(f"Error in generate_text_with_claude: {e}")
        return "Invalid generation"

def generate_text_with_detailed_prompt(x_i, client):
    prompt = (
        "Complete the following sentence by adding words to form a complete sentence ending with a period. "
        "Do not change or rephrase the beginning of the sentence. "
        "Only provide the continuation of the sentence; do not include any additional comments, emoticons, explanations, or notes.\n\n"
        "Examples:\n"
        "Sentence: Yesterday I went\n"
        "Completion: to Costco and purchased a floor cleaner.\n\n"
        "Sentence: She decided to\n"
        "Completion: take a shower.\n\n"
        f"Sentence: {x_i}\n"
        "Completion:"
    )
    return generate_text_with_claude(prompt, client)

def complete_texts_batch(input_texts, client, max_length=100, batch_size=32, max_attempts=5):
    all_completed_texts = []
    
    for i in tqdm(range(0, len(input_texts), batch_size)):
        batch_texts = input_texts[i:i+batch_size]
        valid_completions = ["Invalid generation"] * len(batch_texts)
        
        for attempt in range(max_attempts):
            remaining_indices = [j for j, text in enumerate(valid_completions) if text == "Invalid generation"]
            if not remaining_indices:
                break
            
            current_batch = [batch_texts[j] for j in remaining_indices]
            
            for j, x_i in enumerate(current_batch):
                generated_text = generate_text_with_detailed_prompt(x_i, client)
                cleaned_text = clean_text(generated_text, x_i)
                                
                if is_valid_sentence(cleaned_text, x_i):
                    valid_completions[remaining_indices[j]] = cleaned_text
                else:
                    logging.warning(f"Invalid sentence generated for: {x_i}")
                
                time.sleep(0.5)
        
        all_completed_texts.extend(valid_completions)
    
    return all_completed_texts

def generate_dataset(api_key, input_file, output_file, batch_size=128, max_rows=None):
    client = anthropic.Client(api_key=api_key)
    
    # Read CSV file with option to limit rows
    df = pd.read_csv(input_file)
    if max_rows is not None:
        df = df.head(max_rows)
    
    x_i_samples = df['X_i'].tolist()
    
    x_j_samples = complete_texts_batch(x_i_samples, client, max_length=100, batch_size=batch_size)
    
    results = pd.DataFrame({
        'X_i': x_i_samples,
        'X_j': x_j_samples,
        'LLM': 'Claude-3-haiku'
    })
    
    # Save results to a CSV file
    results.to_csv(output_file, index=False)
    
    return results

# Main function to be called from main.ipynb
def run_claude_generation(api_key, input_file, output_file, batch_size=128, max_rows=None):
    return generate_dataset(api_key, input_file, output_file, batch_size, max_rows)