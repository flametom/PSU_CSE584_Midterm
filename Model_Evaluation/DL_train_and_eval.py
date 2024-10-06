import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from tqdm.auto import tqdm
import joblib

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel
from transformers import RobertaModel, RobertaTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import DebertaModel, DebertaV2Model, DebertaTokenizer, DebertaV2Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from sklearn.utils.class_weight import compute_class_weight
import random
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import re
from sklearn.model_selection import train_test_split

from .load_and_preprocess import preprocess_data, TextDataset, create_data_loaders

def define_model(model_class, num_classes):
    # 모델 인스턴스화
    model = model_class(num_classes=num_classes)
    
    # 모든 레이어 학습 가능하게 설정
    for param in model.parameters():
        param.requires_grad = True
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, patience=3):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
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
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                val_steps +=1
        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve +=1
        
        if epochs_no_improve == patience:
            print(f'Early stopping triggered after {epoch +1} epochs')
            break
    # Best 모델 로드
    model.load_state_dict(torch.load('best_model.pth'))
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device, label_encoder, train_losses, val_losses):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # 정확도 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # F1 스코어 계산
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # 혼동 행렬 생성
    cm = confusion_matrix(all_labels, all_predictions)
    

    
    # 분류 보고서 출력

    
    # 손실 그래프 출력
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    #plt.title('Training and Validation Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.show()
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    model_name = model.__class__.__name__
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

from .Encoder_only_Classifier import BERTClassifier, XLNetClassifier, RoBERTaClassifier, DeBERTaV3Classifier

def train_and_evaluate(model_class, X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, tokenizer, 
                       max_len=128, batch_size=32,
                       num_epochs=20,  patience=5, save_model_flag = False):
    
    #Data Loader created
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, max_len, batch_size
    )
    
    num_classes = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define model
    model = define_model(model_class, num_classes)
    model.to(device)
        
    if isinstance(model, DeBERTaV3Classifier):
        pretrained_model = model.deberta
    elif isinstance(model, BERTClassifier):
        pretrained_model = model.bert
    elif isinstance(model, RoBERTaClassifier):
        pretrained_model = model.roberta
    elif isinstance(model, XLNetClassifier):
        pretrained_model = model.xlnet
    else:
        raise ValueError("Unknown model class")        

    
    # optimizer setting
    # optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    if isinstance(model, DeBERTaV3Classifier): 
        optimizer = AdamW([
            {'params': pretrained_model.embeddings.parameters(), 'lr': 2e-6},
            {'params': pretrained_model.encoder.parameters(), 'lr': 5e-5},
            {'params': model.fc1.parameters(), 'lr': 1e-5},
            {'params': model.fc2.parameters(), 'lr': 1e-5}
        ], weight_decay=0.02)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.02) 
    
    # loss function - Multi Class
    criterion = nn.CrossEntropyLoss()    
    
    # Scheduler
    total_steps = len(train_loader) * num_epochs  # 10 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Train
    train_losses, val_losses = train_model( 
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, patience
    )
    
    # Evaluate
    evaluation_results = evaluate_model(model, test_loader, device, label_encoder, train_losses, val_losses)
        
    # Model Saving
    if save_model_flag:
        model_name = model.__class__.__name__
        save_model(model, model_name, label_encoder, tokenizer, evaluation_results)

    return evaluation_results


def save_model(model, model_name, label_encoder, tokenizer, results):
    # 현재 날짜와 시간을 포함한 폴더 이름 생성
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"saved_models/{model_name}_{current_time}"
    
    # 폴더 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 모델 상태 저장
    torch.save(model.state_dict(), f"{save_dir}/model_state.pth")
    
    # 레이블 인코더 저장
    import joblib
    joblib.dump(label_encoder, f"{save_dir}/label_encoder.joblib")
    
    # 토크나이저 저장
    tokenizer.save_pretrained(save_dir)
    
    # 결과 저장
    with open(f"{save_dir}/results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Trained model and data are saved in {save_dir}.")