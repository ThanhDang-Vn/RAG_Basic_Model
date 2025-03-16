from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_file = "vectordb/vector_data_db"

# load LLM 
def load_LLM(file_model):
	llm = CTransformers(
		model=file_model,
		model_type="llama",
		gpu_layers=0,
		max_new_token=1024,
		temperature=0.01
	)
	return llm

def create_promt(template):
	return PromptTemplate(template=template, input_variables=["context", "question"])

def create_chain(prompt, llm, db):
	return RetrievalQA.from_chain_type(
		llm=llm,
		chain_type="stuff", 
		retriever=db.as_retriever(search_kwargs={"k":3}), # pick 3 most similar 
		# return_resource_documents=False,
		chain_type_kwargs={'prompt':prompt}
	)

def read_vector_db():
	embedding = GPT4AllEmbeddings(model_path="models/all-MiniLM-L6-v2-f16.gguf")
	db = FAISS.load_local(vector_db_file, embeddings=embedding, allow_dangerous_deserialization=True)
	return db 

template = """<|im_start|>system
You are a helpful AI assistant. Please answer the user accurately. If you don't know, answer "I don't know."
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

db = read_vector_db()
llm = load_LLM(model_file)

prompt = create_promt(template=template)
llm_chain = create_chain(prompt=prompt, llm=llm, db=db)
question = input("Prompt: ")
respond = llm_chain.invoke({"query": question})
print(respond)
