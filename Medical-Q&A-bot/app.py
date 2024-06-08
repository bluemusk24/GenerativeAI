from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *                   # this imports everything in prompt.py file
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone

index_name="medical-chatbot"

#Loading the index
docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

# download the quantized llama-ggml model with langchain import above
llm_ggml = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",
    model_file="llama-2-7b-chat.ggmlv3.q4_0.bin",
    config={'max_new_tokens':512, 'temperature':0.8}
) 
 

qa=RetrievalQA.from_chain_type(
    llm=llm_ggml, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


# download the quantized llama-gguf model with langchain import above. this can also be used instead of ggml model
#llm_gguf = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGUF", 
#                    model_file="llama-2-7b-chat.Q4_K_M.gguf",
#                    config={'max_new_tokens':512, 'temperature':0.8})




@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)