# 🤖 RAG Q&A Assistant

A smart PDF question-answering app using Streamlit, LangChain, and Google Gemini AI.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)

## ✨ Features

- 📄 Upload PDF documents
- 💬 Ask questions in natural language
- 🤖 AI-powered answers using Google Gemini
- 🎨 Beautiful, modern UI
- 💾 Conversation history

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/rag-qa-assistant.git
cd rag-qa-assistant
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup API Key

Create `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

Get your free API key: [Google AI Studio](https://makersuite.google.com/app/apikey)

### 4. Run App
```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

## 📦 Requirements

Create `requirements.txt`:
```
streamlit
langchain
langchain-community
langchain-google-genai
langchain-text-splitters
python-dotenv
pypdf
faiss-cpu
sentence-transformers
```

## 🎯 Usage

1. Upload a PDF file
2. Ask questions about the document
3. Get AI-generated answers
4. Adjust settings in sidebar (temperature, chunks)

## 🛠️ Tech Stack

- **Streamlit** - Web interface
- **LangChain** - LLM framework
- **Google Gemini** - AI model
- **FAISS** - Vector search
- **HuggingFace** - Embeddings

## 📁 Project Structure
```
rag-qa-assistant/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .env               # API keys (not tracked)
├── .gitignore         # Git ignore
└── README.md          # This file
```

## ⚙️ Configuration

Customize in sidebar:
- **Temperature**: 0.0 (focused) to 1.0 (creative)
- **Retrieved Chunks**: 1-20 (more = more context)

## 🐛 Troubleshooting

**API key not found?**
- Check `.env` file exists
- Verify `GEMINI_API_KEY` is set
- Restart Streamlit

**Module errors?**
```bash
pip install --upgrade -r requirements.txt
```

## 📝 License

MIT License

## 👤 Author

Your Name - [GitHub](https://github.com/yourusername)

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

⭐ Star this repo if you find it helpful!
```

And here's the `.gitignore`:
```
# Environment
.env
*.env

# Virtual environment
rag-env/
venv/
env/

# Python
__pycache__/
*.pyc
*.pyo

# Temporary files
temp_upload.pdf
*.pdf

# IDE
.vscode/
.idea/

# OS
.DS_Store