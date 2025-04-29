from groq import Groq
from typing import List, Dict, Any
import tiktoken  # You'll need to install this: pip install tiktoken

class QueryProcessor:
    def __init__(self, embedding_generator, vector_db):
        """Initialize with embedding generator and vector database."""
        self.embedding_generator = embedding_generator
        self.vector_db = vector_db
        self.groq_client = Groq()
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"
        # Initialize tokenizer for token counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Using OpenAI's encoding as approximation
        self.max_tokens = 4000  # Leave room for the response, total limit is 6000
    
    def process_query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Process a user query using RAG (Retrieval-Augmented Generation)."""
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        
        # Search for relevant documents
        indices, distances = self.vector_db.search(query_embedding, k=k)
        
        # Retrieve matching documents
        context_docs = []
        for idx in indices:
            doc, metadata = self.vector_db.get_document(idx)
            context_docs.append({"content": doc, "metadata": metadata})
        
        # Create a prompt for the LLM
        prompt = self._create_prompt(query, context_docs)
        
        # Call GROQ API for response generation
        response = self._call_groq_api(prompt)
        
        return {
            "query": query,
            "retrieved_contexts": context_docs,
            "response": response
        }
    
    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Create a prompt for the LLM with the query and context, respecting token limits."""
        prompt_prefix = f"Question: {query}\n\nContext information:\n"
        prompt_suffix = "Based on the provided context, please answer the question. If the information is not in the context, say so and don't make up an answer. If the question asks for specific data analysis, charts, forecasting, or exports, please provide the appropriate response format and logic."
        
        # Calculate tokens for prefix and suffix
        prefix_tokens = len(self.tokenizer.encode(prompt_prefix))
        suffix_tokens = len(self.tokenizer.encode(prompt_suffix))
        
        # Calculate remaining tokens for context
        available_context_tokens = self.max_tokens - prefix_tokens - suffix_tokens
        
        # Truncate or limit context documents to fit within token limit
        context_text = ""
        contexts_used = 0
        
        for i, doc in enumerate(context_docs):
            doc_text = f"Document {i+1}:\n{doc['content']}\n\n"
            doc_tokens = len(self.tokenizer.encode(doc_text))
            
            if len(self.tokenizer.encode(context_text + doc_text)) <= available_context_tokens:
                context_text += doc_text
                contexts_used += 1
            else:
                # If the first document is too large, truncate it
                if i == 0:
                    # Calculate how many tokens we can use from this document
                    tokens_to_use = available_context_tokens - len(self.tokenizer.encode(context_text + f"Document {i+1}:\n"))
                    if tokens_to_use > 0:
                        # Encode document content
                        content_tokens = self.tokenizer.encode(doc['content'])
                        # Truncate to available tokens
                        truncated_tokens = content_tokens[:tokens_to_use]
                        # Decode back to text
                        truncated_content = self.tokenizer.decode(truncated_tokens)
                        
                        context_text += f"Document {i+1}:\n{truncated_content}...(truncated)\n\n"
                        contexts_used += 1
                break
        
        # Assemble final prompt
        prompt = prompt_prefix + context_text + prompt_suffix
        
        # Double-check the token count
        final_token_count = len(self.tokenizer.encode(prompt))
        if final_token_count > self.max_tokens:
            # Last resort truncation
            tokens = self.tokenizer.encode(prompt)[:self.max_tokens]
            prompt = self.tokenizer.decode(tokens)
        
        return prompt
    
    def _call_groq_api(self, prompt: str) -> str:
        """Call GROQ API with the prompt and return the response."""
        completion = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        
        return completion.choices[0].message.content