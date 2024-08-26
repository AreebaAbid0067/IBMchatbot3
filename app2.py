from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import torch
import os
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize Pinecone and models lazily to avoid overhead at startup
vector_store = None
hf_llm = None

def initialize_pinecone():
    global vector_store
    if vector_store is None:
        embeddings = download_hugging_face_embeddings()
        pinecone.init(api_key=PINECONE_API_KEY)
        index = pinecone.Index("mlchatbot")
        vector_store = Pinecone(index=index, embedding=embeddings)

def initialize_model():
    global hf_llm
    if hf_llm is None:
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

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_question = msg
    print(f"Input: {input_question}")

    # Initialize Pinecone and model if not already initialized
    initialize_pinecone()
    initialize_model()

    # Create a retrieval-based QA chain
    qa = RetrievalQA.from_chain_type(
        llm=hf_llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True, 
        chain_type_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["question"])}
    )

    # Run the QA model
    result = qa({"query": input_question})
    response = result.get("result", "Sorry, I don't know the answer.")

    print(f"Response: {response}")
    return str(response)

if __name__ == '__main__':
    app.run(debug=True)
