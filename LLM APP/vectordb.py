import numpy as np
import faiss
from typing import List, Dict, Any, Tuple

class VectorDatabase:
    def __init__(self, dimension: int = 768):
        """Initialize vector database with specified embedding dimension."""
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
        self.metadata = []
        
    def add_documents(self, embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]] = None):
        """Add documents and their embeddings to the database."""
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]
            
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store original documents and metadata
        start_idx = len(self.documents)
        self.documents.extend(documents)
        self.metadata.extend(metadata)
        
        return start_idx, start_idx + len(documents) - 1
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """Search for similar documents based on a query embedding."""
        distances, indices = self.index.search(query_embedding.astype('float32').reshape(1, -1), k)
        
        return indices[0], distances[0]
    
    def get_document(self, idx):
        """Retrieve a document by its index."""
        # Add validation to prevent index out of range errors
        if idx < 0 or idx >= len(self.documents):
            # Return a default value or raise a more descriptive error
            return "Document not found", {"error": "Invalid document index"}
        
        return self.documents[idx], self.metadata[idx]
