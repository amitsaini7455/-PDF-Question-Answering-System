# PDF Question Answering System

## Overview

This project provides a powerful PDF Question Answering (PDF QA) system. It enables users to upload PDF documents and ask questions about their content. The system uses advanced natural language processing (NLP) techniques to retrieve relevant passages from the PDF and generate accurate answers.

## Features

- **PDF Text Extraction**: Extracts text from uploaded PDF files while cleaning and preprocessing it.
- **Text Embedding and Indexing**: Leverages Sentence Transformers and FAISS for efficient similarity search.
- **Question Answering**: Uses a pre-trained RoBERTa model fine-tuned on SQuAD2 for generating answers.
- **Streamlit-based Interface**: Provides an interactive user interface for uploading PDFs, asking questions, and viewing answers with their context and confidence score.
- **Chat History Export**: Allows users to download their Q&A history as a CSV file.

## Technologies Used

- **Python Libraries**:
  - [pypdf](https://pypi.org/project/pypdf): For extracting text from PDF documents.
  - [transformers](https://huggingface.co/docs/transformers/): For tokenization and question-answering model.
  - [sentence-transformers](https://www.sbert.net/): For creating dense embeddings of text passages.
  - [FAISS](https://faiss.ai/): For efficient similarity search and indexing.
  - [Streamlit](https://streamlit.io/): For creating a user-friendly web interface.
- **Models**:
  - Sentence Transformer: `sentence-transformers/all-mpnet-base-v2`
  - Question Answering Model: `deepset/roberta-base-squad2`

## System Workflow

1. **PDF Upload**:
   - Users upload a PDF file using the Streamlit interface.
   - The system extracts text from the PDF, cleans it, and splits it into overlapping passages for processing.

2. **Embedding and Indexing**:
   - Each passage is embedded into a dense vector representation using Sentence Transformers.
   - FAISS is used to create a similarity index for fast retrieval of relevant passages.

3. **Question Answering**:
   - Users input a question related to the uploaded PDF.
   - The system retrieves the most relevant passages based on the question.
   - The RoBERTa QA model processes the question and passages to generate an answer along with a confidence score.

4. **Display and Export**:
   - Answers are displayed in the chat interface with the source passage and confidence score.
   - Users can download their Q&A history as a CSV file.

## File Structure

- `pdf_qa.py`: Core implementation of the PDF QA system.
- `app.py`: Streamlit-based web application to interact with the QA system.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdf-qa-system.git
   cd pdf-qa-system
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit app in your browser.
2. Upload a PDF document.
3. Ask questions related to the content of the PDF.
4. View answers with context and confidence score in the chat interface.
5. Download the Q&A history as a CSV file if needed.

## Example

- **Input Question**: "What are the main features of this project?"
- **Output Answer**: "The project enables PDF text extraction, question answering, and exporting chat history."
- **Source Passage**: "This project provides a powerful PDF Question Answering (PDF QA) system..."
- **Confidence Score**: 9.85

## Limitations

- Performance may vary depending on the quality of the PDF.
- Works best with well-formatted and text-based PDFs.
- Limited to answering questions based on the content of a single uploaded PDF.

## Future Enhancements

- Support for multi-PDF querying.
- Integration with additional QA models for broader functionality.
- Improved support for scanned PDFs using OCR.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Hugging Face for their excellent transformers library.
- Streamlit for simplifying the creation of web apps.
- FAISS for efficient similarity search.

---

Feel free to reach out with any questions or feedback!

