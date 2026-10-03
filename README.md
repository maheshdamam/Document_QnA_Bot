# Document_QnA_Bot

A Streamlit web app that lets you upload a PDF file and ask questions about it. The app reads the PDF, breaks down the text into chunks, creates text embeddings, and uses Google's Gemini API to answer questions based on the document's content.

## How it works
1. **Upload:** You upload a PDF file through the Streamlit interface.
2. **Process:** The app uses `PyPDFLoader` to extract the text and `RecursiveCharacterTextSplitter` to split it into manageable chunks.
3. **Vector Store:** Text chunks are converted into embeddings using `gemini-embedding-2-preview` and stored in an `InMemoryVectorStore`.
4. **Chat:** When you ask a question, the app searches the vector store for matching text chunks and sends them as context to the `gemini-3.8-flash` model to get an accurate answer.
5. **History:** Session state keeps track of the conversation so you can see your chat history.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com
cd Document_QnA_Bot
```

### 2. Set up a virtual environment
```bash
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
```

### 3. Install required packages
```bash
pip install streamlit langchain-core langchain-community langchain-google-genai pypdf python-dotenv
```

### 4. Add your API Key
Create a file named `.env` in the project root folder and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```