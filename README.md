# Retrieval Augmented Generation from PDF

## About
This is a simple program that applies **LangChain**, **RAG (Retrieval-Augmented Generation)**, and **LLM (Large Language Models)** to create a model capable of answer questions based on PDF files  

### How it works:
1. Extracts data from a PDF file and converts it into embeddings
2. Stores the embeddings in a **Vector Database** for efficient retrieval
3. When a user inputs a prompt, the system searches the vector database to find the most relevant information
4. Uses an LLM to generate a response based on the retrieved data 

This approach improves the accuracy of responses by combining **retrieval-based search** with **generative AI** 

## Set up environment

```sh
pip install poetry  
poetry install  
``` 

