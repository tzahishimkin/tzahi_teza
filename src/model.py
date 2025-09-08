"""
RelevanceModel for Crocs RTB bidding system.

Combines sentence transformers with logistic regression for relevance scoring.
Includes brand safety checks, CPM mapping, and caching for performance.
"""

import os
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve


class LRUCache:
    """Simple LRU cache implementation with maximum size limit."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            # Update existing key
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
        self.cache[key] = value


class RelevanceModel:
    """
    Relevance model for Crocs RTB bidding system.
    
    Uses sentence transformers for text encoding and logistic regression
    for relevance classification with CPM-based pricing.
    """
    
    def __init__(self, encoder_name: str = 'sentence-transformers/paraphrase-MiniLM-L3-v2'):
        """
        Initialize the relevance model.
        
        Args:
            encoder_name: Name of the sentence transformer model to use
        """
        self.encoder_name = encoder_name
        self._encoder = None  # lazy init
        self.brief_embedding: Optional[np.ndarray] = None
        self.brief_text: Optional[str] = None
        self.classifier: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold: float = 0.5
        self.min_cpm: float = 0.5
        self.max_cpm: float = 3.0
        
        # Brand safety denylist
        self.denylist = {
            # Violence & Crime
            'violence', 'violent', 'murder', 'kill', 'death', 'assault', 'attack', 'fight',
            'beating', 'torture', 'abuse', 'terrorism', 'terrorist', 'bomb', 'explosion',
            'shooting', 'stabbing', 'crime', 'criminal', 'gang', 'mafia', 'cartel',
            
            # Adult Content
            'pornography', 'porn', 'sex', 'sexual', 'nude', 'nudity', 'explicit', 'adult',
            'xxx', 'erotic', 'escort', 'prostitution', 'strip', 'brothel',
            
            # Gambling & Betting
            'gambling', 'casino', 'poker', 'blackjack', 'slots', 'betting', 'wager',
            'lottery', 'jackpot', 'roulette', 'sportsbook', 'odds', 'bet365',
            
            # Weapons & Military
            'weapon', 'weapons', 'gun', 'guns', 'rifle', 'pistol', 'ammunition', 'bullet',
            'knife', 'sword', 'explosive', 'grenade', 'military', 'war', 'combat',
            
            # Hate Speech & Extremism
            'hate', 'extremist', 'nazi', 'fascist', 'racist', 'racism', 'supremacist',
            'genocide', 'holocaust', 'discrimination', 'bigotry', 'intolerance',
            
            # Drugs & Substances
            'drugs', 'cocaine', 'heroin', 'marijuana', 'cannabis', 'meth', 'addiction',
            'overdose', 'dealer', 'trafficking', 'substance', 'narcotics', 'opium',
            
            # Self-Harm & Suicide
            'suicide', 'self-harm', 'cutting', 'overdose', 'depression', 'mental-illness',
            'self-injury', 'anorexia', 'bulimia', 'eating-disorder',
            
            # Illegal Activities
            'fraud', 'scam', 'theft', 'robbery', 'burglary', 'smuggling', 'counterfeiting',
            'money-laundering', 'bribery', 'corruption', 'hacking', 'piracy',
            
            # Controversial Topics
            'abortion', 'politics', 'election', 'voting', 'political', 'partisan',
            'conspiracy', 'misinformation', 'fake-news',
            
            # Financial Risks
            'pyramid-scheme', 'ponzi', 'cryptocurrency', 'crypto', 'bitcoin', 'investment-scam',
            'get-rich-quick', 'forex', 'trading-scam',
            
            # Health Misinformation
            'miracle-cure', 'alternative-medicine', 'anti-vaccine', 'covid-conspiracy',
            'health-scam', 'weight-loss-scam', 'supplement-scam',
            
            # Inappropriate for Children
            'tobacco', 'smoking', 'vaping', 'alcohol', 'drinking', 'beer', 'wine',
            'liquor', 'mature-content', 'age-restricted'
        }
        
        # Caches
        self.embedding_cache = LRUCache(max_size=1000)
        self.prediction_cache = LRUCache(max_size=1000)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_encoder(self):
        """Get the encoder, creating it lazily if needed."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder
    
    def embed(self, texts):
        """Encode texts using sentence transformer."""
        enc = self._get_encoder()
        vecs = enc.encode(texts, normalize_embeddings=False)
        return np.asarray(vecs, dtype=np.float32)
    
    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using sentence transformer with caching.
        
        Args:
            text: Input text to encode
            
        Returns:
            Text embedding as numpy array
        """
        cache_key = self._get_cache_key(text)
        cached_embedding = self.embedding_cache.get(cache_key)
        
        if cached_embedding is not None:
            return cached_embedding
        
        # Encode text using the new embed method
        embedding = self.embed([text])[0]
        
        # Cache the embedding
        self.embedding_cache.put(cache_key, embedding)
        
        return embedding
    
    def _check_brand_safety(self, snippet: str) -> bool:
        """
        Check if snippet passes brand safety filters.
        
        Args:
            snippet: Text snippet to check
            
        Returns:
            True if safe, False if blocked
        """
        snippet_lower = snippet.lower()
        return not any(word in snippet_lower for word in self.denylist)
    
    def _build_features(self, snippet_embedding: np.ndarray) -> np.ndarray:
        """
        Build feature vector from snippet embedding and brief embedding.
        
        Args:
            snippet_embedding: Encoded snippet
            
        Returns:
            Feature vector combining multiple similarity metrics
        """
        if self.brief_embedding is None:
            raise ValueError("Brief embedding not set. Call fit() first.")
        
        # Compute cosine similarity
        cos_sim = np.dot(snippet_embedding, self.brief_embedding) / (
            np.linalg.norm(snippet_embedding) * np.linalg.norm(self.brief_embedding)
        )
        
        # Element-wise product
        elementwise_product = snippet_embedding * self.brief_embedding
        
        # Absolute difference
        abs_difference = np.abs(snippet_embedding - self.brief_embedding)
        
        # Concatenate all features
        features = np.concatenate([
            snippet_embedding,                    # Original snippet embedding
            [cos_sim],                           # Cosine similarity (scalar)
            elementwise_product,                 # Element-wise product
            abs_difference                       # Absolute difference
        ])
        
        return features
    
    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Find optimal threshold by maximizing F1 score.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Optimal threshold value
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5
        
        return best_threshold
    
    def fit(self, labeled_csv: str, brief_txt: str) -> None:
        """
        Train the relevance model.
        
        Args:
            labeled_csv: Path to CSV file with columns 'snippet' and 'label'
            brief_txt: Crocs creative brief text
        """
        # Store brief text and compute embedding
        self.brief_text = brief_txt
        self.brief_embedding = self._encode_text(brief_txt)
        
        # Load training data
        df = pd.read_csv(labeled_csv)
        snippets = df['snippet'].tolist()
        labels = df['label'].values
        
        # Build feature matrix
        features = []
        for snippet in snippets:
            snippet_embedding = self._encode_text(snippet)
            feature_vector = self._build_features(snippet_embedding)
            features.append(feature_vector)
        
        X = np.array(features)
        y = labels
        
        # Split for threshold optimization
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train logistic regression
        self.classifier = LogisticRegression(random_state=42)
        self.classifier.fit(X_train_scaled, y_train)
        
        # Optimize threshold on validation set
        y_val_prob = self.classifier.predict_proba(X_val_scaled)[:, 1]
        self.threshold = self._optimize_threshold(y_val, y_val_prob)
        
        print(f"Model trained successfully. Optimal threshold: {self.threshold:.3f}")
    
    def save(self, directory: str) -> None:
        """
        Save model to directory using joblib.
        
        Args:
            directory: Directory to save model files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save model components
        model_data = {
            'encoder_name': self.encoder_name,
            'brief_embedding': self.brief_embedding,
            'brief_text': self.brief_text,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'min_cpm': self.min_cpm,
            'max_cpm': self.max_cpm,
            'denylist': self.denylist
        }
        
        joblib.dump(model_data, os.path.join(directory, 'model.pkl'))
        print(f"Model saved to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Load model from directory using joblib.
        
        Args:
            directory: Directory containing saved model files
        """
        model_path = os.path.join(directory, 'model.pkl')
        model_data = joblib.load(model_path)
        
        # Restore model components
        self.encoder_name = model_data['encoder_name']
        self.brief_embedding = model_data['brief_embedding']
        self.brief_text = model_data['brief_text']
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.threshold = model_data['threshold']
        self.min_cpm = model_data['min_cpm']
        self.max_cpm = model_data['max_cpm']
        self.denylist = model_data['denylist']
        
        # Reinitialize encoder lazily (not saved to avoid large file sizes)
        self._encoder = None
        
        print(f"Model loaded from {directory}")
    
    def predict(self, snippet: str) -> Dict[str, float]:
        """
        Predict bid decision and CPM price for a snippet.
        
        Args:
            snippet: Text snippet to evaluate
            
        Returns:
            Dictionary with 'bid', 'price', and 'score' keys
        """
        # Check cache first
        cache_key = self._get_cache_key(snippet)
        cached_prediction = self.prediction_cache.get(cache_key)
        
        if cached_prediction is not None:
            return cached_prediction
        
        # Brand safety check
        if not self._check_brand_safety(snippet):
            result = {'bid': 0, 'price': 0.0, 'score': 0.0}
            self.prediction_cache.put(cache_key, result)
            return result
        
        # Check if model is trained
        if self.classifier is None or self.scaler is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Encode snippet and build features
        snippet_embedding = self._encode_text(snippet)
        features = self._build_features(snippet_embedding).reshape(1, -1)
        
        # Scale features and predict
        features_scaled = self.scaler.transform(features)
        probability = self.classifier.predict_proba(features_scaled)[0, 1]
        
        # Make bid decision
        if probability < self.threshold:
            bid = 0
            price = 0.0
        else:
            bid = 1
            # Map probability to CPM linearly
            price = self.min_cpm + (self.max_cpm - self.min_cpm) * probability
        
        result = {
            'bid': bid,
            'price': float(price),
            'score': float(probability)
        }
        
        # Cache the prediction
        self.prediction_cache.put(cache_key, result)
        
        return result


if __name__ == "__main__":
    # Example usage
    model = RelevanceModel()
    print("RelevanceModel initialized successfully!")
    print(f"Model encoder: {model.encoder_name}")
    print(f"Brand safety denylist: {model.denylist}")
    print(f"Cache sizes - Embeddings: {len(model.embedding_cache.cache)}, Predictions: {len(model.prediction_cache.cache)}")
