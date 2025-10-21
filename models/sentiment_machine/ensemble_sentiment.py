#ensemble_sentiment.py

import re
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from collections import defaultdict

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
