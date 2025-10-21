#deeplearning.py

import tensorflow as tf
from tensorflow.keras import layers, models
import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict

class DLSentimentAnalyzer:
    """
    Analisis sentimen berbasis Deep Learning (TensorFlow/Keras)
    
    Arsitektur:
    - BiLSTM (Bidirectional LSTM)
    - Optional: IndoBERT (coming soon)
    """
    
    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 128,
                 lstm_units: int = 64, max_length: int = 100):
        """
        Initialize DL analyzer
        
        Args:
            vocab_size: Ukuran vocabulary
            embedding_dim: Dimensi embedding
            lstm_units: Jumlah units LSTM
            max_length: Panjang maksimal sequence
        """
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_length = max_length
        
        self.model = None
        self.vectorizer = None
        self.label2id = {"neg": 0, "neu": 1, "pos": 2}
        self.id2label = {0: "Negatif", 1: "Netral", 2: "Positif"}
    
    def clean_text(self, text: str) -> str:
        """Clean text"""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def build_model(self, dropout_rate: float = 0.3):
        """Build BiLSTM model"""

        
        # Text vectorization layer
        self.vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            output_mode="int",
            output_sequence_length=self.max_length,
            standardize=None,
            split="whitespace"
        )
        
        # Model architecture
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text_input")
        x = self.vectorizer(inputs)
        x = layers.Embedding(self.vocab_size, self.embedding_dim)(x)
        x = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))(x)
        x = layers.GlobalMaxPooling1D()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(3, activation="softmax")(x)
        
        self.model = models.Model(inputs, outputs, name="bilstm_sentiment")
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return self.model
    
    def train(self, texts: List[str], labels: List[str], 
              validation_split: float = 0.15, epochs: int = 30,
              batch_size: int = 32, verbose: int = 1):
        """
        Training model
        
        Args:
            texts: List of text
            labels: List of labels
            validation_split: Proporsi validation data
            epochs: Jumlah epochs
            batch_size: Batch size
            verbose: Verbosity
        """
        import tensorflow as tf
        
        if self.model is None:
            self.build_model()
        
        # Clean texts
        texts_clean = [self.clean_text(t) for t in texts]
        
        # Adapt vectorizer
        self.vectorizer.adapt(texts_clean)
        
        # Encode labels
        y = np.array([self.label2id.get(l.lower(), 1) for l in labels])
        
        # Callbacks
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        # Training
        history = self.model.fit(
            x=tf.constant(texts_clean),
            y=y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es, reduce_lr],
            verbose=verbose
        )
        
        return history
    
    def predict(self, texts: Union[str, List[str]]) -> List[Dict]:
        """
        Prediksi sentimen
        
        Args:
            texts: String atau list of strings
        
        Returns:
            List of dict dengan hasil prediksi
        """
        import tensorflow as tf
        
        if self.model is None:
            raise ValueError("Model belum di-train. Gunakan method train() terlebih dahulu.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Clean texts
        texts_clean = [self.clean_text(t) for t in texts]
        
        # Predict
        probs = self.model.predict(
            tf.constant(texts_clean),
            batch_size=32,
            verbose=0
        )
        
        results = []
        for text, prob in zip(texts, probs):
            pred_label = self.id2label[int(np.argmax(prob))]
            results.append({
                'text': text,
                'label': pred_label,
                'confidence': float(np.max(prob)),
                'prob_neg': float(prob[0]),
                'prob_neu': float(prob[1]),
                'prob_pos': float(prob[2])
            })
        
        return results
    
    def evaluate(self, texts: List[str], labels: List[str]) -> Dict:
        """Evaluasi model"""
        import tensorflow as tf
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        texts_clean = [self.clean_text(t) for t in texts]
        y_true = np.array([self.label2id.get(l.lower(), 1) for l in labels])
        
        probs = self.model.predict(tf.constant(texts_clean), verbose=0)
        y_pred = np.argmax(probs, axis=1)
        
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
        """Save model"""
        self.model.save(filepath)
        
        # Save config
        config = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'max_length': self.max_length,
            'label2id': self.label2id,
            'id2label': self.id2label
        }
        
        config_path = filepath.replace('.keras', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load model"""
        import tensorflow as tf
        
        self.model = tf.keras.models.load_model(filepath)
        
        # Load config
        config_path = filepath.replace('.keras', '_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.vocab_size = config['vocab_size']
            self.embedding_dim = config['embedding_dim']
            self.lstm_units = config['lstm_units']
            self.max_length = config['max_length']
            self.label2id = config['label2id']
            self.id2label = {int(k): v for k, v in config['id2label'].items()}
        
        return self
