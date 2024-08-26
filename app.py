from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')



embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
from pinecone import Pinecone

pc = Pinecone(api_key="9ea3155f-be6e-4c0f-aa59-a6ae0d1e19b4")
index = pc.Index("mlchatbot")

#Creating Embeddings for Each of The Text Chunks & storing
from langchain_pinecone import PineconeVectorStore

vector_store = PineconeVectorStore(index=index, embedding=embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["question"])

chain_type_kwargs={"prompt": PROMPT}
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model path
model_path = "ibm-granite/granite-8b-code-instruct"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the model (without quantization)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",  # Automatically choose GPU or CPU based on availability
    low_cpu_mem_usage=True  # This helps reduce memory usage on CPU
)

#context = "IBM website details on cloud computing"
question = "What are the key features of IBM Cloud?"

# Format the prompt template with actual context and question
formatted_prompt = PROMPT.format(question=question)


# Example prompt
#prompt = PROMPT

# Tokenize the input
inputs = tokenizer(formatted_prompt, return_tensors="pt")

# Generate a response
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.5
)

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(response)


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    max_new_tokens=50
    #device=0 if torch.cuda.is_available() else -1
)

# Wrap the pipeline in a LangChain-compatible class
hf_llm = HuggingFacePipeline(pipeline=pipe)

# Create a retrieval-based QA chain
qa = RetrievalQA.from_chain_type(
    llm=hf_llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs
)

# Example prompt for QA
prompt = PROMPT

# Run the QA model
result = qa.invoke(prompt)

# Print the result
#print(result)



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
    app.run(debug=True)

