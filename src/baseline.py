"""
Baseline cosine similarity model for Crocs RTB relevance comparison.

Simple baseline that computes cosine similarity between snippets and the Crocs brief,
with an F1-optimized threshold for binary classification.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


class BaselineCosine:
    """
    Baseline model using simple cosine similarity with the Crocs brief.
    
    Uses the same encoder as the learned model for fair comparison.
    """
    
    def __init__(self, encoder_name: str = 'sentence-transformers/paraphrase-MiniLM-L3-v2'):
        """
        Initialize baseline model.
        
        Args:
            encoder_name: Same encoder as used in RelevanceModel
        """
        self.encoder_name = encoder_name
        self.encoder = SentenceTransformer(encoder_name)
        self.brief_embedding: Optional[np.ndarray] = None
        self.threshold: float = 0.5
    
    def embed(self, texts):
        """Encode texts using sentence transformer (same as RelevanceModel)."""
        vecs = self.encoder.encode(texts, normalize_embeddings=False)
        return np.asarray(vecs, dtype=np.float32)
    
    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    
    def _optimize_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Find optimal threshold by maximizing F1 score."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        
        return best_threshold
    
    def fit(self, labeled_csv: str, brief_text: str) -> None:
        """
        Train the baseline by encoding the brief and optimizing threshold.
        
        Args:
            labeled_csv: Path to training data with 'snippet' and 'label' columns
            brief_text: Crocs campaign brief text
        """
        # Encode the brief once (using same embed method as RelevanceModel)
        self.brief_embedding = self.embed([brief_text])[0]
        
        # Load training data
        df = pd.read_csv(labeled_csv)
        snippets = df['snippet'].tolist()
        labels = df['label'].values
        
        # Compute cosine similarities for all training examples
        similarities = []
        for snippet in snippets:
            snippet_embedding = self.embed([snippet])[0]
            similarity = self._compute_cosine_similarity(snippet_embedding, self.brief_embedding)
            similarities.append(similarity)
        
        similarities = np.array(similarities)
        
        # Split train/val same as RelevanceModel (80/20, stratified, random_state=42)
        # to avoid optimizing threshold on test data
        similarities_train, similarities_val, y_train, y_val = train_test_split(
            similarities, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Optimize threshold on validation set only
        self.threshold = self._optimize_threshold(y_val, similarities_val)
        
        print(f"Baseline trained. Optimal threshold: {self.threshold:.3f} (optimized on validation set)")
    
    def predict(self, snippet: str) -> float:
        """
        Predict cosine similarity score for a snippet.
        
        Args:
            snippet: Text snippet to score
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if self.brief_embedding is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Encode snippet (using same embed method as RelevanceModel)
        snippet_embedding = self.embed([snippet])[0]
        
        # Compute cosine similarity
        similarity = self._compute_cosine_similarity(snippet_embedding, self.brief_embedding)
        
        # Ensure score is between 0 and 1
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def predict_binary(self, snippet: str) -> int:
        """
        Predict binary relevance (0/1) for a snippet.
        
        Args:
            snippet: Text snippet to classify
            
        Returns:
            1 if relevant (similarity > threshold), 0 otherwise
        """
        score = self.predict(snippet)
        return 1 if score > self.threshold else 0


if __name__ == "__main__":
    # Example usage
    baseline = BaselineCosine()
    print("BaselineCosine initialized successfully!")
    print(f"Encoder: {baseline.encoder_name}")
