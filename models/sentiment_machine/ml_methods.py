#ml_methods.py

import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict

class MLSentimentAnalyzer:
    """
    Analisis sentimen berbasis Machine Learning (Sklearn)
    
    Algoritma yang didukung:
    - Naive Bayes
    - Linear SVM
    - Logistic Regression
    - Random Forest
    """
    
    def __init__(self):
        """Initialize ML analyzer"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = None
        self.model = None
        self.label2id = {"neg": 0, "neu": 1, "pos": 2}
        self.id2label = {0: "Negatif", 1: "Netral", 2: "Positif"}
        
        # Stopwords Indonesia
        self.stopwords = {
            "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan",
            "atau", "itu", "ini", "saya", "kami", "kita", "kamu", "anda",
            "dia", "ia", "mereka", "ada", "jadi", "aja", "kok", "lah",
            "pun", "akan", "udah", "sudah", "belum", "telah", "dalam",
            "agar", "karena", "sebagai", "bahwa", "juga", "tidak", "bukan",
            "yg", "nya", "si"
        }
    
    def preprocess_text(self, text: str) -> str:
        """Preprocessing teks"""
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"[0-9]+", " ", text)
        text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        # Remove stopwords
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        
        return " ".join(tokens)
    
    def train(self, texts: List[str], labels: List[str], 
              model_type: str = "svm", **kwargs):
        """
        Training model
        
        Args:
            texts: List of text
            labels: List of labels ('pos', 'neg', 'neu')
            model_type: 'naive_bayes', 'svm', 'logistic', 'random_forest'
            **kwargs: Parameter tambahan untuk model
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.ensemble import RandomForestClassifier
        
        # Vectorization
        self.vectorizer = TfidfVectorizer(
            preprocessor=self.preprocess_text,
            tokenizer=lambda s: s.split(),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=1000
        )
        
        X = self.vectorizer.fit_transform(texts)
        
        # Encode labels
        y = np.array([self.label2id.get(l.lower(), 1) for l in labels])
        
        # Select model
        models = {
            'naive_bayes': MultinomialNB,
            'svm': LinearSVC,
            'logistic': LogisticRegression,
            'random_forest': RandomForestClassifier
        }
        
        if model_type not in models:
            raise ValueError(f"Model type must be one of {list(models.keys())}")
        
        # Default parameters
        default_params = {
            'naive_bayes': {'alpha': 1.0},
            'svm': {'C': 1.0, 'random_state': 42, 'max_iter': 2000},
            'logistic': {'max_iter': 500, 'random_state': 42},
            'random_forest': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        }
        
        params = {**default_params[model_type], **kwargs}
        
        self.model = models[model_type](**params)
        self.model.fit(X, y)
        
        return self
    
    def predict(self, texts: Union[str, List[str]]) -> List[Dict]:
        """
        Prediksi sentimen
        
        Args:
            texts: String atau list of strings
        
        Returns:
            List of dict dengan hasil prediksi
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model belum di-train. Gunakan method train() terlebih dahulu.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        
        results = []
        for text, pred in zip(texts, predictions):
            result = {
                'text': text,
                'label': self.id2label[pred]
            }
            
            # Add probability if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X)[len(results)]
                result['confidence'] = float(max(proba))
                result['prob_neg'] = float(proba[0])
                result['prob_neu'] = float(proba[1])
                result['prob_pos'] = float(proba[2])
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X)[len(results)]
                exp_scores = np.exp(scores - np.max(scores))
                proba = exp_scores / exp_scores.sum()
                result['confidence'] = float(max(proba))
            
            results.append(result)
        
        return results
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        """
        Evaluasi model
        
        Returns:
            Dict dengan metrics: accuracy, precision, recall, f1
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        X = self.vectorizer.transform(texts)
        y_true = np.array([self.label2id.get(l.lower(), 1) for l in labels])
        y_pred = self.model.predict(X)
        
        acc = accuracy_score(y_true, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        return {
            'accuracy': round(acc, 4),
            'precision': round(pr, 4),
            'recall': round(rc, 4),
            'f1_score': round(f1, 4)
        }
    
    def save_model(self, filepath: str):
        """Save model dan vectorizer"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f)
    
    def load_model(self, filepath: str):
        """Load model dan vectorizer"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.label2id = data['label2id']
            self.id2label = data['id2label']
        return self

