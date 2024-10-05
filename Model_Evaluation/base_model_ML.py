from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import spacy
from typing import List, Dict
import numpy as np
import pandas as pd

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder):
    models = {
        'MNB': MultinomialNB(),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'LR': LogisticRegression(multi_class='auto', max_iter=2000),
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(),
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} model:")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)    
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        
        results[name] = {
            'accuracy': accuracy,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'report': report
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}")
        print(f"{name} - F1: {f1:.4f}")
        # print(f"{name} - Confusion Matrix:\n{conf_matrix}")
        print(f"{name} - Classification Report:\n{report}") 

        # Inside the train_and_evaluate_models function, add the following code after computing `conf_matrix`:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix for {name} Model')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    return results

def plot_accuracies(results: Dict):
    names = list(results.keys())
    accuracies = [result['accuracy'] for result in results.values()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, accuracies)
    plt.title('Model Accuracies')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    plt.show()