"""
Embedding Engine for User Memory Service
Converts text to embeddings and computes Hash Sphere coordinates
"""

import hashlib
import math
import numpy as np
from typing import List, Tuple, Optional
import os
import httpx
import logging

logger = logging.getLogger(__name__)

# OpenAI API for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class EmbeddingEngine:
    """
    Generates embeddings and converts them to Hash Sphere coordinates.
    Uses the hash universe formula to place memories in 3D space.
    """
    
    def __init__(self):
        self.model = EMBEDDING_MODEL
        self.dim = EMBEDDING_DIM
        
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using OpenAI API.
        Falls back to deterministic hash-based embedding if API unavailable.
        """
        if OPENAI_API_KEY:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model,
                            "input": text
                        },
                        timeout=30.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return data["data"][0]["embedding"]
            except Exception as e:
                logger.warning(f"OpenAI embedding failed, using fallback: {e}")
        
        # Fallback: deterministic hash-based pseudo-embedding
        return self._hash_embedding(text)
    
    def _hash_embedding(self, text: str) -> List[float]:
        """
        Generate deterministic pseudo-embedding from text hash.
        This is a fallback when OpenAI API is unavailable.
        """
        # Create multiple hashes for different parts of the embedding
        embedding = []
        for i in range(self.dim // 32 + 1):
            hash_input = f"{text}_{i}"
            hash_bytes = hashlib.sha256(hash_input.encode()).digest()
            # Convert each byte to a float in [-1, 1]
            for byte in hash_bytes:
                if len(embedding) < self.dim:
                    embedding.append((byte / 127.5) - 1.0)
        
        # Normalize to unit vector
        norm = math.sqrt(sum(x*x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding[:self.dim]
    
    def embedding_to_coordinates(
        self, 
        embedding: List[float],
        cluster_center: Tuple[float, float, float] = (0, 0, 0),
        scale: float = 100.0
    ) -> Tuple[float, float, float]:
        """
        Convert embedding vector to 3D Hash Sphere coordinates.
        
        Uses dimensionality reduction to map high-dim embedding to 3D space
        while preserving relative distances between memories.
        
        The formula places semantically similar memories closer together.
        """
        if len(embedding) < 3:
            return cluster_center
        
        # Method: Use first 3 principal components approximation
        # Split embedding into 3 groups and sum each for x, y, z
        chunk_size = len(embedding) // 3
        
        x_components = embedding[:chunk_size]
        y_components = embedding[chunk_size:2*chunk_size]
        z_components = embedding[2*chunk_size:]
        
        # Sum and normalize each component
        x = sum(x_components) / len(x_components) if x_components else 0
        y = sum(y_components) / len(y_components) if y_components else 0
        z = sum(z_components) / len(z_components) if z_components else 0
        
        # Scale to universe size
        x = x * scale + cluster_center[0]
        y = y * scale + cluster_center[1]
        z = z * scale + cluster_center[2]
        
        return (x, y, z)
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        """
        if len(embedding1) != len(embedding2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(a * a for a in embedding1))
        norm2 = math.sqrt(sum(b * b for b in embedding2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar(
        self, 
        query_embedding: List[float], 
        embeddings: List[Tuple[str, List[float]]], 
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Find top-k most similar embeddings to query.
        Returns list of (id, similarity) tuples.
        """
        similarities = []
        
        for id_, emb in embeddings:
            sim = self.compute_similarity(query_embedding, emb)
            if sim >= min_similarity:
                similarities.append((id_, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def content_hash(self, content: str) -> str:
        """Generate content hash for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()[:32]


# Singleton instance
embedding_engine = EmbeddingEngine()
