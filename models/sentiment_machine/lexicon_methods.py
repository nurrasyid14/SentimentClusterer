#lexicon_methods

import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict

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
