Tiny RAG
========

This is a very minimal retrieval augmented generation (RAG)
implementation. Supply your own documents and ask questions.

You need an OpenAI API key from <https://platform.openai.com/api-keys>.
Define that API key as environment variable `OPENAI_API_KEY`. Then

```bash
# Install dependencies.
pip install -U click dataiter jinja2 numpy openai pypdf2

# If you're stuck with NumPy 1.x, try
# pip install -U click dataiter==0.51 jinja2 openai pypdf2

# Add PDF documents to the RAG database.
./tinyrag.py -p *.pdf

# Ask a question based on the created RAG database.
./tinyrag.py -q "What is the difference between null and void?"
```
