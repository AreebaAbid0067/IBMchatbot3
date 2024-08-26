from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import time
from src.prompt2 import prompt_template2

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

# Global variables
vector_store = None
hf_llm = None

def initialize_services():
    global vector_store, hf_llm
    if vector_store is None:
        print("Initializing Pinecone...")
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        index_name = "mlchatbot"
        index = pc.Index(index_name)
        embeddings = download_hugging_face_embeddings()
        vector_store = Pinecone(index, embeddings.embed_query, "text")

    if hf_llm is None:
        print("Initializing Model...")
        model_path = "ibm-granite/granite-8b-code-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        pipe = pipeline(
            "text-generation", 
            model=model, 
            tokenizer=tokenizer,
            max_new_tokens=50
        )
        hf_llm = HuggingFacePipeline(pipeline=pipe)

# Initialize Pinecone and model once at startup
initialize_services()

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    start_time = time.time()
    msg = request.form["msg"]
    input_question = msg
    print(f"Input Question: {input_question}")

    # Create a retrieval-based QA chain
    qa = RetrievalQA.from_chain_type(
        llm=hf_llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True, 
        chain_type_kwargs={
            "prompt": PromptTemplate(template=prompt_template2, input_variables=["context", "question"])
        }
    )

    result = qa({"query": input_question})
    response = result.get("result", "Sorry, I don't know the answer.")
    response = response.split('Answer:')[-1].strip()

    end_time = time.time()
    print(f"Response: {response}")
    print(f"Response Time: {end_time - start_time} seconds")
    return str(response)

if __name__ == '__main__':
    app.run(debug=True)
