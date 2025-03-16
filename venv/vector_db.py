from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# PyPDFLoader use for load data from PDF file 
# DirectoryLoader for scan data from PDF file 
data_path = "data"
vector_stored = "vectordb/vector_data_db"

def pdf_to_vector_db():
	loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
	document = loader.load()

	text_spliter = RecursiveCharacterTextSplitter(
		chunk_size = 512, 
		chunk_overlap = 50	
	)

	chunk = text_spliter.split_documents(documents=document)
	embedding = GPT4AllEmbeddings(model_path="models/all-MiniLM-L6-v2-f16.gguf")

	db = FAISS.from_documents(documents=chunk, embedding=embedding)
	db.save_local(vector_stored)
	return db


pdf_to_vector_db()
	

