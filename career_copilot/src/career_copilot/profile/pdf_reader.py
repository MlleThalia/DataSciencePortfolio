import pdfplumber
from fastapi import UploadFile
import logging

def read_pdf(file_path: UploadFile) -> str:
    """
    Reads the text content from a PDF file.

    Args:
        file_path (UploadFile): The uploaded PDF file.
    
    """
    text = ""
    with pdfplumber.open(file_path.file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n\n"
    
    logging.info(f"Text extracted from PDF: {len(text)} characters")
    return text