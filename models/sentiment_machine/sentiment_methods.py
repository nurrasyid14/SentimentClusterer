"""
========================================
SENTIMENT ANALYSIS LIBRARY - MODULAR
========================================
Library lengkap untuk analisis sentimen dengan 3 pendekatan:
1. Lexicon-based (Rule-based)
2. Machine Learning (Sklearn)
3. Deep Learning (TensorFlow/Keras + IndoBERT)

Author: Sentiment Analysis Team
Date: 2025
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict

# =========================================================
# MODULE 1: LEXICON-BASED SENTIMENT ANALYSIS
# =========================================================

class LexiconSentimentAnalyzer:
    """
    Analisis sentimen berbasis lexicon (kamus kata)
    
    Fitur:
    - Normalisasi teks
    - Negation handling
    - Intensifier/downtoner
    - Emoji support
    - Custom lexicon
    """
    
    def __init__(self, custom_lexicon: Optional[Dict[str, float]] = None):
        """
        Initialize analyzer
        
        Args:
            custom_lexicon: Dict kata -> skor sentimen
        """
        # Default lexicon Bahasa Indonesia
        self.lexicon = {
            # Positif umum
            "bagus": 2.0, "mantap": 2.5, "enak": 2.0, "puas": 2.0, 
            "cepat": 1.5, "top": 2.5, "recommended": 2.0, "ramah": 1.8,
            "suka": 1.8, "keren": 1.8, "hebat": 2.2,
            
            # Positif sepakbola
            "menang": 2.0, "juara": 2.5, "terbaik": 2.3, "berkualitas": 2.0,
            "prestasi": 1.8, "lolos": 2.0, "sukses": 2.2, "dukungan": 1.5,
            "kredibel": 1.5, "penghargaan": 1.8, "pemenang": 2.0,
            
            # Negatif umum
            "buruk": -2.5, "mahal": -1.5, "lambat": -1.8, "kecewa": -2.2,
            "jelek": -2.2, "parah": -2.0, "mengecewakan": -2.5,
            
            # Negatif sepakbola
            "kalah": -2.0, "gagal": -2.3, "blunder": -2.5, "merosot": -2.2,
            "terpuruk": -2.5, "hancur": -2.8, "salah": -1.8, "dipecat": -2.0,
            "capek": -1.7,
            
            # Netral
            "biasa": 0.0, "lumayan": 0.5, "oke": 0.8, "standar": 0.0
        }
        
        # Update dengan custom lexicon jika ada
        if custom_lexicon:
            self.lexicon.update(custom_lexicon)
        
        # Intensifiers & Downtoners
        self.intensifiers = {
            "banget": 1.5, "sekali": 1.4, "sangat": 1.6, "amat": 1.4,
            "terlalu": 1.4, "lagi": 1.3, "terus": 1.2, "langsung": 1.3
        }
        
        self.downtoners = {
            "agak": 0.7, "cukup": 0.8, "lumayan": 0.85, "sepertinya": 0.7
        }
        
        # Negation words
        self.negations = {
            "tidak", "tak", "bukan", "nggak", "ga", "gak",
            "tiada", "jangan", "belum", "tanpa"
        }
        
        # Emoji mapping
        self.emoji_map = {
            "😀": 1.5, "😃": 1.5, "😄": 1.5, "😁": 1.5, "🙂": 1.5,
            "😊": 1.5, "👍": 1.5, "❤": 1.5, "❤️": 1.5,
            "😞": -1.5, "😡": -1.5, "😠": -1.5, "☹": -1.5,
            "😢": -1.5, "💔": -1.5, "👎": -1.5,
            ":(": -1.5, ":-(": -1.5, ":)": 1.5, ":-)": 1.5
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalisasi teks"""
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenisasi dengan preservasi emoji"""
        # Preservasi emoji
        for emoji in sorted(self.emoji_map.keys(), key=len, reverse=True):
            text = text.replace(emoji, f" {emoji} ")
        
        tokens = re.findall(r"[a-zA-Z0-9]+|[^\s\w]", text)
        return [t for t in tokens if t.strip()]
    
    def score_text(self, text: str, window_neg: int = 3) -> Tuple[float, str, Dict]:
        """
        Hitung skor sentimen
        
        Args:
            text: Input text
            window_neg: Window size untuk negation detection
        
        Returns:
            (score, label, detail)
        """
        normalized = self.normalize_text(text)
        tokens = self.tokenize(normalized)
        
        if not tokens:
            return 0.0, "Netral", {}
        
        # Detect emoji
        emoji_score = sum(self.emoji_map.get(t, 0) for t in tokens)
        
        # Detect negation indices
        neg_idx = [i for i, t in enumerate(tokens) if t in self.negations]
        
        # Score per token
        token_scores = [0.0] * len(tokens)
        detail = {}
        
        for i, token in enumerate(tokens):
            if token in self.lexicon:
                base = self.lexicon[token]
                factor = 1.0
                
                # Check intensifier/downtoner
                if i > 0:
                    prev = tokens[i - 1]
                    if prev in self.intensifiers:
                        factor *= self.intensifiers[prev]
                    if prev in self.downtoners:
                        factor *= self.downtoners[prev]
                
                if i < len(tokens) - 1:
                    next_tok = tokens[i + 1]
                    if next_tok in self.intensifiers:
                        factor *= self.intensifiers[next_tok]
                    if next_tok in self.downtoners:
                        factor *= self.downtoners[next_tok]
                
                # Check negation
                is_negated = any(j < i and (i - j) <= window_neg for j in neg_idx)
                if is_negated:
                    base = -base
                
                token_scores[i] = base * factor
                if token_scores[i] != 0:
                    detail[token] = round(token_scores[i], 3)
        
        # Total score
        score = sum(token_scores) + emoji_score
        
        # Exclamation boost
        exclaim_count = text.count("!")
        if exclaim_count > 0:
            score *= min(1.0 + 0.05 * exclaim_count, 1.3)
        
        # Label
        if score > 0.8:
            label = "Positif"
        elif score < -0.8:
            label = "Negatif"
        else:
            label = "Netral"
        
        if emoji_score != 0:
            detail["__emoji__"] = round(emoji_score, 3)
        
        return round(score, 3), label, detail
    
    def analyze_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """
        Analisis sentimen untuk DataFrame
        
        Args:
            df: DataFrame dengan kolom text
            text_col: Nama kolom yang berisi teks
        
        Returns:
            DataFrame dengan kolom tambahan: score, predicted_label, detail
        """
        results = []
        for text in df[text_col]:
            score, label, detail = self.score_text(str(text))
            results.append({
                'score': score,
                'predicted_label': label,
                'detail': detail
            })
        
        result_df = pd.concat([df, pd.DataFrame(results)], axis=1)
        return result_df
    
    def add_word(self, word: str, score: float):
        """Tambah kata ke lexicon"""
        self.lexicon[word.lower().strip()] = float(score)
    
    def save_lexicon(self, filepath: str):
        """Save lexicon ke file JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.lexicon, f, ensure_ascii=False, indent=2)
    
    def load_lexicon(self, filepath: str):
        """Load lexicon dari file JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            custom = json.load(f)
            self.lexicon.update(custom)


# =========================================================
# MODULE 2: MACHINE LEARNING SENTIMENT ANALYSIS
# =========================================================

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


# =========================================================
# MODULE 3: DEEP LEARNING SENTIMENT ANALYSIS
# =========================================================

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
        import tensorflow as tf
        
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
        import tensorflow as tf
        from tensorflow.keras import layers, models
        
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


# =========================================================
# MODULE 4: UTILITY FUNCTIONS
# =========================================================

def load_json_data(filepath: str) -> pd.DataFrame:
    """
    Load data dari JSON file
    
    Args:
        filepath: Path ke file JSON
    
    Returns:
        DataFrame
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Rename dan mapping label jika perlu
    if 'sentiment_label' in df.columns:
        df = df.rename(columns={'sentiment_label': 'label'})
        
        label_mapping = {
            'Positif': 'pos',
            'Negatif': 'neg',
            'Netral': 'neu'
        }
        df['label'] = df['label'].map(label_mapping)
    
    return df


def evaluate_model(y_true: List[str], y_pred: List[str], 
                   labels: List[str] = None) -> Dict:
    """
    Evaluasi comprehensive untuk model
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: List of label names
    
    Returns:
        Dict dengan metrics lengkap
    """
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, precision_recall_fscore_support)
    
    if labels is None:
        labels = ["neg", "neu", "pos"]
    
    # Normalize labels
    y_true_norm = [l.lower() if isinstance(l, str) else l for l in y_true]
    y_pred_norm = [l.lower() if isinstance(l, str) else l for l in y_pred]
    
    # Metrics
    acc = accuracy_score(y_true_norm, y_pred_norm)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true_norm, y_pred_norm, average='macro', zero_division=0
    )
    
    cm = confusion_matrix(y_true_norm, y_pred_norm, labels=labels)
    
    report = classification_report(
        y_true_norm, y_pred_norm,
        labels=labels,
        target_names=[l.upper() for l in labels],
        digits=3
    )
    
    return {
        'accuracy': round(acc, 4),
        'precision_macro': round(pr, 4),
        'recall_macro': round(rc, 4),
        'f1_macro': round(f1, 4),
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_confusion_matrix(cm: np.ndarray, labels: List[str],
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        labels: List of label names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[l.upper() for l in labels],
        yticklabels=[l.upper() for l in labels],
        ax=ax, cbar_kws={'label': 'Count'}
    )
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.show()


# =========================================================
# MODULE 5: ENSEMBLE ANALYZER
# =========================================================

class EnsembleSentimentAnalyzer:
    """
    Ensemble dari multiple analyzers dengan voting
    """
    
    def __init__(self):
        """Initialize ensemble"""
        self.analyzers = {}
        self.weights = {}
    
    def add_analyzer(self, name: str, analyzer, weight: float = 1.0):
        """
        Tambah analyzer ke ensemble
        
        Args:
            name: Nama analyzer
            analyzer: Instance dari analyzer (Lexicon/ML/DL)
            weight: Bobot untuk voting (default: 1.0)
        """
        self.analyzers[name] = analyzer
        self.weights[name] = weight
    
    def predict(self, texts: Union[str, List[str]], 
                method: str = "weighted_voting") -> List[Dict]:
        """
        Prediksi dengan ensemble
        
        Args:
            texts: String atau list of strings
            method: 'weighted_voting', 'majority_voting', 'average_prob'
        
        Returns:
            List of dict dengan hasil prediksi ensemble
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.analyzers:
            raise ValueError("Belum ada analyzer. Gunakan add_analyzer() terlebih dahulu.")
        
        # Collect predictions dari semua analyzer
        all_predictions = {}
        for name, analyzer in self.analyzers.items():
            try:
                if isinstance(analyzer, LexiconSentimentAnalyzer):
                    preds = []
                    for text in texts:
                        score, label, _ = analyzer.score_text(text)
                        # Convert label ke format standar
                        label_map = {"Positif": "pos", "Negatif": "neg", "Netral": "neu"}
                        preds.append({
                            'label': label_map.get(label, "neu"),
                            'score': score
                        })
                    all_predictions[name] = preds
                else:
                    # ML atau DL analyzer
                    preds = analyzer.predict(texts)
                    # Normalize label format
                    for pred in preds:
                        label = pred.get('label', 'neu')
                        if isinstance(label, str):
                            label = label.lower()
                            if label in ['positif', 'positive']:
                                pred['label'] = 'pos'
                            elif label in ['negatif', 'negative']:
                                pred['label'] = 'neg'
                            else:
                                pred['label'] = 'neu'
                    all_predictions[name] = preds
            except Exception as e:
                print(f"Warning: {name} prediction failed: {e}")
                continue
        
        # Ensemble predictions
        results = []
        label_to_id = {"neg": 0, "neu": 1, "pos": 2}
        id_to_label = {0: "Negatif", 1: "Netral", 2: "Positif"}
        
        for i in range(len(texts)):
            if method == "weighted_voting":
                # Weighted voting
                votes = defaultdict(float)
                for name, preds in all_predictions.items():
                    label = preds[i]['label']
                    votes[label] += self.weights[name]
                
                final_label = max(votes.items(), key=lambda x: x[1])[0]
                confidence = votes[final_label] / sum(votes.values())
                
            elif method == "majority_voting":
                # Simple majority
                votes = [preds[i]['label'] for preds in all_predictions.values()]
                final_label = max(set(votes), key=votes.count)
                confidence = votes.count(final_label) / len(votes)
                
            elif method == "average_prob":
                # Average probabilities jika ada
                probs = np.zeros(3)
                count = 0
                
                for name, preds in all_predictions.items():
                    pred = preds[i]
                    if 'prob_neg' in pred:
                        probs[0] += pred['prob_neg'] * self.weights[name]
                        probs[1] += pred['prob_neu'] * self.weights[name]
                        probs[2] += pred['prob_pos'] * self.weights[name]
                        count += self.weights[name]
                    else:
                        # Convert label to one-hot
                        label_id = label_to_id[pred['label']]
                        probs[label_id] += self.weights[name]
                        count += self.weights[name]
                
                if count > 0:
                    probs /= count
                    final_label_id = np.argmax(probs)
                    final_label = list(label_to_id.keys())[final_label_id]
                    confidence = probs[final_label_id]
                else:
                    final_label = "neu"
                    confidence = 0.33
            else:
                raise ValueError(f"Method {method} tidak dikenal")
            
            # Format label
            label_map = {"pos": "Positif", "neg": "Negatif", "neu": "Netral"}
            
            results.append({
                'text': texts[i],
                'label': label_map.get(final_label, "Netral"),
                'confidence': float(confidence),
                'individual_predictions': {
                    name: preds[i]['label'] 
                    for name, preds in all_predictions.items()
                }
            })
        
        return results


# =========================================================
# EXAMPLE USAGE & DEMO
# =========================================================

def demo_usage():
    """
    Demo penggunaan library
    """
    print("="*80)
    print("SENTIMENT ANALYSIS LIBRARY - DEMO")
    print("="*80)
    
    # Sample data
    sample_texts = [
        "Timnas Indonesia menang! Pelatih sangat bagus!",
        "Kalah lagi, sangat mengecewakan sekali",
        "Pertandingannya biasa saja, tidak terlalu istimewa"
    ]
    
    sample_labels = ["pos", "neg", "neu"]
    
    # ========== 1. LEXICON-BASED ==========
    print("\n" + "="*80)
    print("1. LEXICON-BASED ANALYZER")
    print("="*80)
    
    lexicon_analyzer = LexiconSentimentAnalyzer()
    
    for text in sample_texts:
        score, label, detail = lexicon_analyzer.score_text(text)
        print(f"\nTeks: {text}")
        print(f"Skor: {score}")
        print(f"Label: {label}")
        print(f"Detail: {detail}")
    
    # ========== 2. MACHINE LEARNING ==========
    print("\n" + "="*80)
    print("2. MACHINE LEARNING ANALYZER")
    print("="*80)
    
    # Create more training data
    train_texts = sample_texts * 10  # Duplicate untuk demo
    train_labels = sample_labels * 10
    
    ml_analyzer = MLSentimentAnalyzer()
    ml_analyzer.train(train_texts, train_labels, model_type='svm')
    
    predictions = ml_analyzer.predict(sample_texts)
    for pred in predictions:
        print(f"\nTeks: {pred['text']}")
        print(f"Label: {pred['label']}")
        if 'confidence' in pred:
            print(f"Confidence: {pred['confidence']:.2%}")
    
    # ========== 3. DEEP LEARNING ==========
    print("\n" + "="*80)
    print("3. DEEP LEARNING ANALYZER (BiLSTM)")
    print("="*80)
    print("Note: Training DL membutuhkan lebih banyak data dan waktu")
    print("Demo dilewati. Gunakan dengan data yang lebih besar.")
    
    # ========== 4. ENSEMBLE ==========
    print("\n" + "="*80)
    print("4. ENSEMBLE ANALYZER")
    print("="*80)
    
    ensemble = EnsembleSentimentAnalyzer()
    ensemble.add_analyzer("lexicon", lexicon_analyzer, weight=0.3)
    ensemble.add_analyzer("ml_svm", ml_analyzer, weight=0.7)
    
    ensemble_preds = ensemble.predict(sample_texts, method="weighted_voting")
    for pred in ensemble_preds:
        print(f"\nTeks: {pred['text']}")
        print(f"Label: {pred['label']}")
        print(f"Confidence: {pred['confidence']:.2%}")
        print(f"Individual: {pred['individual_predictions']}")
    
    print("\n" + "="*80)
    print("✨ DEMO SELESAI!")
    print("="*80)


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    print("""
    ========================================
    SENTIMENT ANALYSIS LIBRARY
    ========================================
    
    Available Classes:
    1. LexiconSentimentAnalyzer - Rule-based sentiment analysis
    2. MLSentimentAnalyzer - Machine learning sentiment analysis
    3. DLSentimentAnalyzer - Deep learning sentiment analysis
    4. EnsembleSentimentAnalyzer - Ensemble multiple analyzers
    
    Utility Functions:
    - load_json_data() - Load data dari JSON
    - evaluate_model() - Evaluasi model
    - plot_confusion_matrix() - Plot confusion matrix
    
    Untuk melihat demo penggunaan, jalankan:
    >>> demo_usage()
    """)
    
    # Uncomment untuk menjalankan demo
    # demo_usage()