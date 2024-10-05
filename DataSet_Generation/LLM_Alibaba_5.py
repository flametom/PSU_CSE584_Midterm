import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import re
import time

def clean_text(text, x_i):
    text = re.sub(r'\s+', ' ', text).strip()
    x_i = re.sub(r'\s+', ' ', x_i).strip()
    
    if text.lower().startswith(x_i.lower()):
        text = text[len(x_i):].lstrip()
        
    x_i_index = text.lower().find(x_i.lower())
    if x_i_index != -1:
        text = text[x_i_index + len(x_i):].lstrip()

    sentences = re.findall(r'[^.!?]+[.!?]', text)
    
    if len(sentences) < 5:
        return "Invalid generation"
    
    selected_sentences = sentences[:5]
    
    cleaned_sentences = []
    for sentence in selected_sentences:
        sentence = sentence.strip()
        sentence = re.sub(r'\s+([.!?])', r'\1', sentence)
        sentence = re.sub(r'([.!?])\1+$', r'\1', sentence)
        cleaned_sentences.append(sentence)
    
    final_text = x_i + " " + ' '.join(cleaned_sentences)
    
    return final_text

def is_valid_sentence(text, x_i):
    if not text or text == "Invalid generation":
        return False

    if not text.lower().replace(" ", "").startswith(x_i.lower().replace(" ", "")):
        return False

    remaining_text = text[len(x_i):].strip()

    sentences = re.findall(r'[^.!?]+[.!?]', remaining_text)
    if len(sentences) != 5:
        return False

    words = remaining_text.split()
    if len(words) < 15 or len(words) > 250:
        return False

    if '_' in text:
        return False

    unwanted_phrases = [
        'Text:', 'Solution:', 'Answer:', 'B:', 'Note', 'example:', 'Sentence:', 'Completion:',
        'Here is your text:', 'Note that', '(Note' , 'your turn', 'Human:', 'Assistant:'
    ]
    for phrase in unwanted_phrases:
        if phrase.lower() in text.lower():
            return False

    if text.count('"') % 2 != 0:
        return False

    if has_repetition(words):
        return False

    return True

def has_repetition(words):
    for i in range(len(words) - 2):
        if words[i:i+3] == words[i+3:i+6]:
            return True
    return False

def generate_text(prompt, model, tokenizer, max_length=200):
    messages = [
        {"role": "system", "content": "Complete the following sentence by adding words to form exactly five complete sentences, including the initial one. Do not change or rephrase the beginning of the first sentence. Ensure that the five sentences are coherent and flow logically. Each sentence should end with a period, exclamation mark, or question mark. Only provide the continuation of the text; do not include any additional comments, emoticons, explanations, or notes."},
        {"role": "user", "content": "Sentence: Yesterday I went"},
        {"role": "assistant", "content": "to Costco and purchased a floor cleaner. The store was bustling with shoppers looking for deals. I spent an hour browsing through various aisles. In addition to the floor cleaner, I picked up some groceries and household items. Overall, it was a productive shopping trip."},
        {"role": "user", "content": f"Sentence: {prompt}"}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def complete_texts_batch(input_texts, model, tokenizer, max_length=200, batch_size=4, max_attempts=5):
    all_completed_texts = []

    for i in tqdm(range(0, len(input_texts), batch_size)):
        batch_texts = input_texts[i:i+batch_size]
        valid_completions = ["Invalid generation"] * len(batch_texts)

        for attempt in range(max_attempts):
            remaining_indices = [j for j, text in enumerate(valid_completions) if text == "Invalid generation"]
            if not remaining_indices:
                break

            current_batch = [batch_texts[j] for j in remaining_indices]

            for idx, x_i in enumerate(current_batch):
                generated_text = generate_text(x_i, model, tokenizer, max_length)
                cleaned_text = clean_text(generated_text, x_i)

                if cleaned_text != "Invalid generation" and is_valid_sentence(cleaned_text, x_i):
                    valid_completions[remaining_indices[idx]] = cleaned_text

                time.sleep(0.5)

        all_completed_texts.extend(valid_completions)

    return all_completed_texts

def generate_dataset_5(input_file, output_file, model_name="Qwen/Qwen2.5-3B-Instruct", batch_size=32, max_rows=None):
    torch.manual_seed(20241006)

    df = pd.read_csv(input_file)
    if max_rows is not None:
        df = df.head(max_rows)
    
    x_i_samples = df['X_i'].tolist()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    x_j_samples = complete_texts_batch(x_i_samples, model, tokenizer, max_length=200, batch_size=batch_size)

    results = pd.DataFrame({
        'X_i': x_i_samples,
        'X_j': x_j_samples,
        'LLM': 'Qwen2.5-3B-Instruct'
    })

    results.to_csv(output_file, index=False)
    
    return results

# Main function to be called from main.ipynb
def run_qwen_generation_5(input_file, output_file, model_name="Qwen/Qwen2.5-3B-Instruct", batch_size=32, max_rows=None):
    return generate_dataset_5(input_file, output_file, model_name, batch_size, max_rows)