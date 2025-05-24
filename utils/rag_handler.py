import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L12-v2')

def save_index(chunks, pdf_name, chunk_to_page_map):
    """
    Create and save FAISS index for document chunks.
    
    Args:
        chunks: List of text chunks
        pdf_name: Name of the PDF file (for folder naming)
        chunk_to_page_map: Mapping of chunks to their source page numbers
        
    Returns:
        str: Path to the saved index folder
    """
    # Create embeddings for chunks
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Create directory for storing the index
    index_dir = os.path.join('data', f"{pdf_name}_faiss")
    os.makedirs(index_dir, exist_ok=True)
    
    # Save the index
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    
    # Save the chunks and mapping for later retrieval
    with open(os.path.join(index_dir, "chunks.pkl"), 'wb') as f:
        pickle.dump(chunks, f)
        
    with open(os.path.join(index_dir, "chunk_to_page_map.pkl"), 'wb') as f:
        pickle.dump(chunk_to_page_map, f)
    
    return index_dir

def load_index(index_dir):
    """
    Load FAISS index and related data from a directory.
    
    Args:
        index_dir: Path to the directory containing the index
        
    Returns:
        tuple: (faiss_index, chunks, chunk_to_page_map)
    """
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    
    with open(os.path.join(index_dir, "chunks.pkl"), 'rb') as f:
        chunks = pickle.load(f)
        
    with open(os.path.join(index_dir, "chunk_to_page_map.pkl"), 'rb') as f:
        chunk_to_page_map = pickle.load(f)
    
    return index, chunks, chunk_to_page_map

def retrieve_chunks(query, index, chunks, chunk_to_page_map, top_k=5):
    """
    Retrieve the most relevant chunks for a query.
    
    Args:
        query: User query string
        index: FAISS index
        chunks: List of text chunks
        chunk_to_page_map: Mapping of chunks to their source page numbers
        top_k: Number of chunks to retrieve
        
    Returns:
        tuple: (retrieved_chunks, page_numbers)
    """
    # Create query embedding
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search the index
    distances, indices = index.search(query_embedding, top_k)
    
    # Get the retrieved chunks and their page numbers
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    page_numbers = []
    
    for chunk in retrieved_chunks:
        if chunk in chunk_to_page_map:
            for page in chunk_to_page_map[chunk]:
                if page not in page_numbers:
                    page_numbers.append(page)
    
    return retrieved_chunks, page_numbers