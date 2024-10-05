import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import os
from datetime import datetime
from .DL_train_and_eval import save_model
from .load_and_preprocess import preprocess_data, TextDataset
from tqdm.auto import tqdm

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        rnn_output, _ = self.rnn(embedded)
        pooled = self.global_max_pool(rnn_output.transpose(1, 2)).squeeze(2)
        x = self.dropout1(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.softmax(self.fc2(x), dim=1)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_output, _ = self.lstm(embedded)
        pooled = self.global_max_pool(lstm_output.transpose(1, 2)).squeeze(2)
        x = self.dropout1(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.softmax(self.fc2(x), dim=1)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate=0.3):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        gru_output, _ = self.gru(embedded)
        pooled = self.global_max_pool(gru_output.transpose(1, 2)).squeeze(2)
        x = self.dropout1(pooled)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return F.softmax(self.fc2(x), dim=1)
    
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # * 2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)
    
class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_filters=100, filter_sizes=(3, 4, 5), num_layers=2, dropout=0.3):
        super(CNNBiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.lstm = nn.LSTM(len(filter_sizes) * num_filters, hidden_dim, num_layers=num_layers, 
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids).unsqueeze(1)
        conv_outputs = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled_outputs = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in conv_outputs]
        cnn_output = torch.cat(pooled_outputs, 1)
        lstm_output, (hidden, cell) = self.lstm(cnn_output.unsqueeze(1))
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

def train_model_NN(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=50,  patience=5, val_split=0.1):
    
    # Splitting the train_loader into train and validation
    train_size = int((1 - val_split) * len(train_loader.dataset))
    val_size = len(train_loader.dataset) - train_size
    train_dataset, val_dataset = random_split(train_loader.dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=train_loader.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=train_loader.batch_size, num_workers=train_loader.num_workers)

    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 그래디언트 클리핑 적용
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            # scheduler.step()

            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                # attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {total_val_loss/len(val_loader):.4f}')
        

        if scheduler:
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    return train_losses, val_losses

def train_and_evaluate_NN(model_class, X_train, X_test, y_train, y_test, label_encoder, tokenizer, 
                       embed_dim=256, hidden_dim=512, max_len=128, batch_size=32,
                       num_epochs=50,  patience=5, val_split=0.1, save_model_flag = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    train_dataset = TextDataset(X_train, y_train, tokenizer, max_len)
    test_dataset = TextDataset(X_test, y_test, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    num_classes = len(label_encoder.classes_)
    vocab_size = tokenizer.vocab_size  


    model = model_class(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.02)  # weight decay 추가
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    train_losses, val_losses = train_model_NN(model, train_loader, criterion, optimizer, scheduler, device, num_epochs,  patience, val_split)

    evaluation_results = evaluate_model_NN(model, test_loader, device, label_encoder, train_losses, val_losses)
    
    # Model Saving
    if save_model_flag:
        model_name = model.__class__.__name__
        save_model(model, model_name, label_encoder, tokenizer, evaluation_results)

    return evaluation_results

def evaluate_model_NN(model, test_loader, device, label_encoder, train_losses, val_losses):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids)
            _, predictions = torch.max(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    

    accuracy = accuracy_score(all_labels, all_predictions)
    

    f1 = f1_score(all_labels, all_predictions, average='weighted')

    cm = confusion_matrix(all_labels, all_predictions)
    
    model_name = model.__class__.__name__
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}")
    print(f"{model_name} - F1: {f1:.4f}")
    
    class_names = label_encoder.classes_
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=label_encoder.classes_,
               yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return results







