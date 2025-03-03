DocuMind AI

## Overview
DocuMind AI is a Streamlit-based intelligent document processing and question-answering system that allows users to upload a PDF document, extract questions from the text, and receive AI-generated answers based on the document's context. The application utilizes Langchain, Google Gemini AI, FAISS for vector storage, and LangGraph to manage workflows.

## Features
- Upload a PDF file
- Automatically extract questions from the document
- Retrieve relevant context using FAISS
- Generate answers using Google Gemini AI
- Display thought process for each answer
- Allow users to manually ask additional questions
- Reset and process new PDFs


https://github.com/user-attachments/assets/0272a0cb-364b-458f-8109-4d352fbde87e


## Working Video


## Tech Stack
- **Frontend:** Streamlit  
- **AI Models:** Google Gemini (Generative AI)  
- **Text Processing:** Langchain, PyPDFLoader  
- **Vector Storage:** FAISS  
- **Workflow Management:** LangGraph  
- **Environment Management:** dotenv  

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed.

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your API keys and configuration values:
     ```env
     GOOGLE_API_KEY=<your-google-api-key>
     ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload a PDF file.
2. The app will extract questions automatically.
3. Review the extracted questions and their AI-generated answers.
4. Ask additional questions about the document manually.
5. Reset and upload a new PDF if needed.

## Directory Structure
```
📂 project-directory
├── app.py               # Main application script
├── requirements.txt     # Required dependencies
├── .env                 # Environment variables (not included in repo)
└── README.md            # Documentation
```

## Dependencies
- streamlit
- langchain
- langgraph
- faiss-cpu
- python-dotenv
- PyPDFLoader

## License
This project is licensed under the MIT License.

## Author
Gagan N

