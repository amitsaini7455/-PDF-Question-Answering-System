# pdf_qa.py

import pypdf
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Dict
import faiss
import re

class PDFQuestionAnswering:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the PDF Question Answering system
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        self.index = None
        self.passages = []
        
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text from PDF
        Args:
            text: Raw text from PDF
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        
        # Remove special characters and markdown-like syntax
        text = re.sub(r'[~*_]{2,}', '', text)
        
        # Remove extra punctuation
        text = re.sub(r'[:,]\s*[:,]', ':', text)
        
        return text.strip()

    # Modify the extract_text_from_pdf method:
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        Args:
            pdf_path: Path to PDF file
        Returns:
            Extracted text from PDF
        """
        pdf_reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += self.clean_text(page_text) + "\n"
        return text
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text into passages
        Args:
            text: Input text
        Returns:
            List of text passages
        """
        # Split into sentences and create overlapping passages
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
        
        passages = []
        window_size = 3
        stride = 2
        
        for i in range(0, len(sentences), stride):
            passage = " ".join(sentences[i:i + window_size])
            if len(passage.split()) >= 10:  # Only keep passages with at least 10 words
                passages.append(passage)
                
        return passages
    
    def create_embeddings_index(self, passages: List[str]):
        """
        Create FAISS index for fast similarity search
        Args:
            passages: List of text passages
        """
        self.passages = passages
        embeddings = self.embedding_model.encode(passages)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
    
    def process_pdf(self, pdf_path: str):
        """
        Process PDF file and create embeddings index
        Args:
            pdf_path: Path to PDF file
        """
        text = self.extract_text_from_pdf(pdf_path)
        passages = self.preprocess_text(text)
        self.create_embeddings_index(passages)
    
    def get_relevant_passages(self, query: str, k: int = 3) -> List[str]:
        """
        Get most relevant passages for a query
        Args:
            query: Question to find relevant passages for
            k: Number of passages to retrieve
        Returns:
            List of relevant passages
        """
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        return [self.passages[i] for i in indices[0]]
    
    def answer_question(self, question: str) -> Dict[str, str]:
        """
        Answer a question using the processed PDF content
        Args:
            question: Question to answer
        Returns:
            Dictionary containing answer and source passage
        """
        if not self.index:
            return {"error": "No PDF has been processed yet"}
        
        relevant_passages = self.get_relevant_passages(question)
        best_answer = ""
        best_score = 0
        best_passage = ""
        
        for passage in relevant_passages:
            inputs = self.qa_tokenizer(
                question,
                passage,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
            
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits)
            
            if answer_end >= answer_start:
                answer = self.qa_tokenizer.convert_tokens_to_string(
                    self.qa_tokenizer.convert_ids_to_tokens(
                        inputs["input_ids"][0][answer_start:answer_end + 1]
                    )
                )
                
                score = float(torch.max(outputs.start_logits)) + float(torch.max(outputs.end_logits))
                
                if score > best_score and len(answer.strip()) > 0:
                    best_score = score
                    best_answer = answer
                    best_passage = passage
        
        return {
            "answer": best_answer,
            "source_passage": best_passage,
            "confidence_score": best_score
        }