import PyPDF2

def extract_text(pdf_file):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_file: The uploaded PDF file object
        
    Returns:
        list: A list of strings, each containing text from a page
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pages = []
    
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pages.append(page.extract_text())
        
    return pages

def chunk_text(pages, chunk_size=1000, overlap=200):
    """
    Split text into chunks with overlap for better context preservation.
    
    Args:
        pages: List of page texts
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        list: List of text chunks
        dict: Mapping of chunks to their source page numbers
    """
    chunks = []
    chunk_to_page_map = {}
    
    for page_num, page_text in enumerate(pages):
        if not page_text.strip():
            continue
            
        # Process each page into overlapping chunks
        start = 0
        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            chunk = page_text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
                # Store the page number (1-indexed) for this chunk
                if chunk not in chunk_to_page_map:
                    chunk_to_page_map[chunk] = []
                chunk_to_page_map[chunk].append(str(page_num + 1))
                
            start += chunk_size - overlap
            if start >= len(page_text):
                break
    
    return chunks, chunk_to_page_map