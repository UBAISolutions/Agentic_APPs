import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """Initialize with a pretrained model for generating embeddings."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text chunks."""
        embeddings = []
        
        for text in texts:
            # Tokenize and prepare for the model
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use mean pooling to get a single vector per text
            embeddings.append(self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask']).cpu().numpy()[0])
        
        return np.array(embeddings)
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling to get a single vector per text."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
