# sentiment_utils.py

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Tuple, Dict
import json
import numpy as np

def load_json_data(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if 'sentiment_label' in df.columns:
        df = df.rename(columns={'sentiment_label': 'label'})
        label_mapping = {'Positif': 'pos', 'Negatif': 'neg', 'Netral': 'neu'}
        df['label'] = df['label'].map(label_mapping)
    return df


def evaluate_model(y_true: List[str], y_pred: List[str], 
                   labels: List[str] = None) -> Dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    
    if labels is None:
        labels = ["neg", "neu", "pos"]
    
    y_true_norm = [l.lower() if isinstance(l, str) else l for l in y_true]
    y_pred_norm = [l.lower() if isinstance(l, str) else l for l in y_pred]
    
    acc = accuracy_score(y_true_norm, y_pred_norm)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true_norm, y_pred_norm, average='macro', zero_division=0)
    cm = confusion_matrix(y_true_norm, y_pred_norm, labels=labels)
    report = classification_report(y_true_norm, y_pred_norm, labels=labels, target_names=[l.upper() for l in labels], digits=3)
    
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
                          save_path: Optional[str] = None):
    """
    Plot interactive confusion matrix using Plotly
    """
    z_text = [[str(int(y)) for y in x] for x in cm]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[l.upper() for l in labels],
        y=[l.upper() for l in labels],
        text=z_text,
        texttemplate="%{text}",
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        width=700,
        height=600
    )
    
    if save_path:
        fig.write_image(save_path)
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    fig.show()
