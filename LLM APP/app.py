# Import all modules
from document import DocumentProcessor
from embedding import EmbeddingGenerator
from vectordb import VectorDatabase
from query import QueryProcessor
from analytics import AnalyticsEngine

# Import required libraries for web interface
from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import base64
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class LLMApplication:
    def __init__(self):
        """Initialize the LLM application with all required components."""
        self.document_processor = DocumentProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        self.query_processor = QueryProcessor(self.embedding_generator, self.vector_db)
        self.analytics_engine = AnalyticsEngine()
        self.loaded_files = {}
        self.loaded_data = {}
        
    def load_document(self, file_path):
        """Load a document into the application."""
        # Process document
        document_content = self.document_processor.process_document(file_path)
        
        # Generate embeddings for the chunks
        embeddings = self.embedding_generator.generate_embeddings(document_content['chunks'])
        
        # Create metadata for vector database
        metadata = [{
            "source_file": os.path.basename(file_path),
            "chunk_index": i,
            "file_type": document_content['type']
        } for i in range(len(document_content['chunks']))]
        
        # Add to vector database
        start_idx, end_idx = self.vector_db.add_documents(
            embeddings, document_content['chunks'], metadata
        )
        
        # Store file info
        file_name = os.path.basename(file_path)
        self.loaded_files[file_name] = {
            "path": file_path,
            "content": document_content,
            "vector_indices": (start_idx, end_idx)
        }
        
        # Store actual data for tabular files
        if document_content['type'] == 'tabular' and 'data' in document_content:
            self.loaded_data[file_name] = document_content['data']
        
        return {
            "file_name": file_name,
            "chunks": len(document_content['chunks']),
            "vector_indices": (start_idx, end_idx),
            "status": "loaded"
        }
    
    def process_query(self, query):
        """Process a user query against loaded documents."""
        # Check if query is likely asking for analysis/visualization
        analysis_keywords = ["chart", "plot", "graph", "forecast", "predict", 
                            "excel", "export", "trend", "pattern", "analyze", 
                            "summarize", "statistics", "correlation"]
        
        is_analysis_query = any(keyword in query.lower() for keyword in analysis_keywords)
        
        if is_analysis_query and self.loaded_data:
            # For analysis queries, we need to find the relevant data file
            response = self.query_processor.process_query(
                f"Which data file should be used to answer this query: {query}?"
            )
            
            file_names = list(self.loaded_data.keys())
            most_relevant_file = file_names[0]  # Default to first file
            
            # Try to extract file name from response
            for file_name in file_names:
                if file_name.lower() in response['response'].lower():
                    most_relevant_file = file_name
                    break
            
            # Perform data analysis on the selected file
            analysis_result = self.analytics_engine.analyze_data(
                self.loaded_data[most_relevant_file], query
            )
            
            # Combine with LLM response for explanation
            final_response = {
                "query": query,
                "analysis": analysis_result,
                "explanation": self.query_processor.process_query(
                    f"Explain the following analysis results for the query '{query}': {analysis_result}"
                )['response'],
                "file_used": most_relevant_file
            }
            
            return final_response
        else:
            # For regular text queries, just use the query processor
            return self.query_processor.process_query(query)
    
    def list_loaded_files(self):
        """List all files loaded into the application."""
        return [{
            "file_name": file_name,
            "file_type": info["content"]["type"],
            "chunks": len(info["content"]["chunks"]),
            "vector_indices": info["vector_indices"]
        } for file_name, info in self.loaded_files.items()]

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Create application instance
llm_app = LLMApplication()

@app.route('/')
def index():
    """Render the home page."""
    loaded_files = llm_app.list_loaded_files()
    return render_template('index.html', files=loaded_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load the document
        result = llm_app.load_document(filepath)
        
        return jsonify(result)

@app.route('/query', methods=['POST'])
def process_query():
    """Process a query."""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data['query']
    result = llm_app.process_query(query)
    
    # Handle different response types
    if 'analysis' in result and 'response_type' in result['analysis']:
        response_type = result['analysis']['response_type']
        
        if response_type in ['chart', 'forecast', 'trend']:
            # For visualization types, the image is already base64 encoded
            pass
        elif response_type == 'export':
            # For exports, data is already base64 encoded
            pass
    
    return jsonify(result)

@app.route('/files')
def list_files():
    """List loaded files."""
    files = llm_app.list_loaded_files()
    return jsonify(files)

if __name__ == '__main__':
    app.run(debug=True)
