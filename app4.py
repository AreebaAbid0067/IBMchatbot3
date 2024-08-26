from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from src.prompt2 import prompt_template2

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')  # Pinecone environment (e.g., 'us-west1-gcp')

vector_store = None
hf_llm = None

def initialize_pinecone():
    global vector_store
    if vector_store is None:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        index_name = "mlchatbot"
        index = pc.Index(index_name)
        embeddings = download_hugging_face_embeddings()
        vector_store = Pinecone(index, embeddings.embed_query, "text")

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

    initialize_pinecone()
    initialize_model()

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
    print(f"Result: {result}")
    
    # Extracting the response
    response = result.get("result", "Sorry, I don't know the answer.")
    
    # Clean up the response to ensure only the answer is returned
    response = response.split('Answer:')[-1].strip()

    print(f"Response: {response}")
    return str(response)

if __name__ == '__main__':
    app.run(debug=True)
