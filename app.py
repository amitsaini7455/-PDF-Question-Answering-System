# app.py

import streamlit as st
import time
import pandas as pd
from pathlib import Path
from pdf_qa import PDFQuestionAnswering  # Import the previous PDF QA class

# Set page configuration
st.set_page_config(
    page_title="PDF Question Answering System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        background-color: white;
        color: black; /* Default text color for all messages */
    }
    .user-message {
        background-color: #f8f9fa;
        border-left: 5px solid #2196F3;
        color: black; /* Ensure user message text is black */
    }
    .bot-message {
        background-color: #ffffff;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: black; /* Ensure bot message text is black */
    }
    .metadata {
        font-size: 0.9rem;
        color: #555; /* Metadata remains a softer color */
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid #eee;
    }
    strong {
        color: black; /* Ensure strong tags are also black */
    }
    </style>
    """, unsafe_allow_html=True)



# Initialize session state variables
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None
if 'question_input' not in st.session_state:
    st.session_state.question_input = ""

def initialize_qa_system():
    """Initialize the QA system with a progress bar"""
    with st.spinner('Initializing QA System...'):
        st.session_state.qa_system = PDFQuestionAnswering()
        st.success('System initialized successfully!')

def process_pdf(pdf_file):
    """Process uploaded PDF file"""
    with st.spinner('Processing PDF... This may take a few minutes.'):
        # Save uploaded file temporarily
        pdf_path = Path('temp_pdf.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(pdf_file.getvalue())
        
        # Process the PDF
        st.session_state.qa_system.process_pdf(str(pdf_path))
        st.session_state.current_pdf = pdf_file.name
        
        # Clean up
        pdf_path.unlink()
        st.success('PDF processed successfully!')

def display_chat_history():
    """Display chat history with custom styling"""
    for message in st.session_state.chat_history:
        if message['type'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>Question:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
        else:
            # Clean the source passage for display
            source = message['source'].replace('~', '').replace('*', '').strip()
            answer = message['content'].replace('~', '').replace('*', '').strip()
            
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Answer:</strong> {answer}
                    <div class="metadata">
                        <div style="margin-top: 8px;"><strong>Source Context:</strong> {source}</div>
                        <div style="margin-top: 4px;"><strong>Confidence Score:</strong> {message['confidence']:.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.question_input = ""

def handle_question():
    if st.session_state.question_input:
        question = st.session_state.question_input
        
        # Add user question to chat history
        st.session_state.chat_history.append({
            'type': 'user',
            'content': question
        })
        
        # Get answer from QA system
        with st.spinner('Thinking...'):
            result = st.session_state.qa_system.answer_question(question)
        
        # Add answer to chat history
        st.session_state.chat_history.append({
            'type': 'assistant',
            'content': result['answer'],
            'source': result['source_passage'],
            'confidence': result['confidence_score']
        })
        
        # Clear the input
        st.session_state.question_input = ""

def main():
    # Header
    st.title("📚 PDF Question Answering System")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload PDF")
        pdf_file = st.file_uploader("Choose a PDF file", type=['pdf'])
        
        if pdf_file and (st.session_state.current_pdf != pdf_file.name):
            if st.session_state.qa_system is None:
                initialize_qa_system()
            process_pdf(pdf_file)
            clear_chat()
        
        if st.session_state.current_pdf:
            st.success(f"Current PDF: {st.session_state.current_pdf}")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Upload a PDF document first
        - Ask clear, specific questions
        - Questions should be related to the PDF content
        - System works best with well-formatted PDFs
        """)
        
        if st.button("Clear Chat History"):
            clear_chat()

    # Main chat interface
    if st.session_state.qa_system and st.session_state.current_pdf:
        # Display chat history
        display_chat_history()
        
        # Question input
        st.text_input(
            "Ask a question about the PDF:",
            key="question_input",
            on_change=handle_question
        )
        
        # Export chat history
        if st.session_state.chat_history:
            chat_data = []
            for i in range(0, len(st.session_state.chat_history), 2):
                if i + 1 < len(st.session_state.chat_history):
                    chat_data.append({
                        'Question': st.session_state.chat_history[i]['content'],
                        'Answer': st.session_state.chat_history[i+1]['content'],
                        'Source': st.session_state.chat_history[i+1]['source'],
                        'Confidence': st.session_state.chat_history[i+1]['confidence']
                    })
            
            df = pd.DataFrame(chat_data)
            st.download_button(
                label="Download Chat History",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='chat_history.csv',
                mime='text/csv'
            )
    else:
        st.info("👈 Please upload a PDF file to start asking questions.")

if __name__ == "__main__":
    main()