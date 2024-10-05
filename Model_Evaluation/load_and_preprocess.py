import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

def is_english(text):

    return bool(re.match(r'^[a-zA-Z0-9\s.,!?"\-\'`$%&*+<=>@^_:;|~àáâäãåąčćęèéêëėįìíîïłńòóôöõøùúûüųūÿýżźñçčšžÀÁÂÄÃÅĄĆČĖĘÈÉÊËÌÍÎÏĮŁŃÒÓÔÖÕØÙÚÛÜŲŪŸÝŻŹÑßÇŒÆČŠŽ∂ð]+$', text))
    
def load_and_preprocess_data(file_paths):
    all_data = []
    for path in file_paths:
        df = pd.read_csv(path, dtype=str)
        if 'X_j' not in df.columns:
            df = pd.read_csv(path, quoting=3, dtype=str)
        all_data.append(df)
    
    combined_df_raw = pd.concat(all_data, ignore_index=True)
    combined_df_raw = combined_df_raw[combined_df_raw['X_j'] != "Invalid generation"]
    conbined_df_raw = combined_df_raw[combined_df_raw['X_i'] != 'maybe a b * * w j * b while wearing']
    
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        
        # Handle specific endings
        endings_to_fix = {
            '.".' : '."',
            "'." : "'",
            ',.' : '.',
            '`.' : '`',
            '".' : '"',
            ').' : '.',
        }
        for old_end, new_end in endings_to_fix.items():
            if text.endswith(old_end):
                text = text[:-len(old_end)] + new_end
        # text = text.lower()
        # text = re.sub(r'[^\w\s]', '', text)
                
        return text.strip()   


    combined_df_raw['processed_text'] = combined_df_raw['X_j'].apply(preprocess_text)
    
    combined_df = combined_df_raw[combined_df_raw['processed_text'].apply(is_english)]
    combined_df_temp = combined_df_raw[~combined_df_raw['processed_text'].apply(is_english)]
    
    le = LabelEncoder()
    combined_df['LLM_encoded'] = le.fit_transform(combined_df['LLM'])
    
    X = combined_df['processed_text'].tolist()
    y = combined_df['LLM_encoded'].tolist()
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=20241006, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=20241006, stratify=y_train_raw)
    
    train_counts = pd.Series(y_train).value_counts()
    val_counts = pd.Series(y_val).value_counts()
    test_counts = pd.Series(y_test).value_counts()
    
    
    print("Sample counts for each LLM model:")
    for llm in le.classes_:
        llm_index = le.transform([llm])[0]
        print(f"{llm}:")
        # print(f"  Total: {train_counts.get(llm_index, 0) + val_counts.get(llm_index, 0) + test_counts.get(llm_index, 0)}")
        print(f"  Train: {train_counts.get(llm_index, 0)}")
        print(f"  Val: {val_counts.get(llm_index, 0)}")
        print(f"  Test:  {test_counts.get(llm_index, 0)}")
        print()
    
    return X_train, X_test, X_val, y_train, y_test, y_val, le, combined_df, combined_df_temp

def preprocess_data(X, y):

    X = pd.Series(X)
    y = pd.Series(y)
    
    df = pd.DataFrame({'text': X, 'label': y})
    df = df.dropna().reset_index(drop=True)
    
    return df['text'].tolist(), df['label'].tolist()

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = pd.Series(texts).reset_index(drop=True)
        self.labels = pd.Series(labels).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts.iloc[item])
        label = self.labels.iloc[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, max_len=128, batch_size=32):

    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)
    X_test, y_test = preprocess_data(X_test, y_test)
    
    train_dataset = TextDataset(X_train, y_train, tokenizer, max_len)
    val_dataset = TextDataset(X_val, y_val, tokenizer, max_len)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)
    
    return train_loader, val_loader, test_loader


from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import spacy
from typing import List, Dict
import numpy as np
import pandas as pd

# Download necessary NLTK data
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class AdvancedTokenizer:
    def __init__(self, tokenizer_type='nltk'):
        self.tokenizer_type = tokenizer_type
        self.word_index = {'<PAD>': 0, '<UNK>': 1}
        self.index_word = {0: '<PAD>', 1: '<UNK>'}
        self.num_words = 2

    def fit_on_texts(self, texts: List[str]):
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.word_index:
                    self.word_index[token] = self.num_words
                    self.index_word[self.num_words] = token
                    self.num_words += 1

    def tokenize(self, text: str) -> List[str]:
        if self.tokenizer_type == 'nltk':
            return nltk.word_tokenize(text)
        elif self.tokenizer_type == 'spacy':
            return [token.text for token in nlp(text)]
        else:
            raise ValueError("Invalid tokenizer type. Choose 'nltk' or 'spacy'.")

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        return [[self.word_index.get(token, self.word_index['<UNK>']) for token in self.tokenize(text)] for text in texts]

def prepare_data_for_ml(X_train, X_test, tokenizer: AdvancedTokenizer):
    # Tokenization and TF-IDF vectorization
    tfidf = TfidfVectorizer(tokenizer=tokenizer.tokenize)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf
    