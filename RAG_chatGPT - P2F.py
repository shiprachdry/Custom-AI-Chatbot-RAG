import json
from ipywidgets import widgets
import os
import requests
import faiss
import pandas as pd
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as sl


with open('C:\\Users\\I068117\\UT_Machine Learning\\Text-Analytics_NLP\\Sentiment Analysis\\irpa-r701-genaixl-customer-fd-sk.txt') as f:
    sk = json.load(f)
    
# Replace this with the resource group provided
resource_group = widgets.Text(
    value='e98ee3d7-45fa-43cb-8773-fb106ed18d67', # resource group
    placeholder='Resource group of deployments',
    description='',
    disabled=False
)
#resource_group

os.environ['AICORE_LLM_AUTH_URL'] = sk['url']+"/oauth/token"
os.environ['AICORE_LLM_CLIENT_ID'] = sk['clientid']
os.environ['AICORE_LLM_CLIENT_SECRET'] = sk['clientsecret']
os.environ['AICORE_LLM_API_BASE'] = sk["serviceurls"]["AI_API_URL"]+ "/v2"
os.environ['AICORE_LLM_RESOURCE_GROUP'] = resource_group.value
os.environ['LLM_COMMONS_PROXY'] = 'aicore'

response = requests.post(
        f'{os.environ["AICORE_LLM_AUTH_URL"]}/oauth/token',
        data={"grant_type": "client_credentials"},
        auth=(os.environ['AICORE_LLM_CLIENT_ID'], os.environ['AICORE_LLM_CLIENT_SECRET']),
        timeout=8000,
)
auth_token = response.json()["access_token"]

def RAG_response_api(context, question):
    review=f"You need to answer the question in the sentence using the context provided in the sentence. Answer like an HR professional.Given below is the context and question of the user. context: {context} question:{question}. if the answer is not in the pdf , answer i donot know what the hell you are asking about.Provide the answer in bullets"
   
    #review=f"You need to answer the question in the sentence as same as in the pdf content.Given below is the context and question of the user. context: {context} question:{question}. if the answer is not in the pdf , answer i donot know what the hell you are asking about"
    test_input = {
        "model" : "gpt-35-turbo-16k",
        "messages" : [{ "content": review,"role": "user"}],
        "max_tokens": 800
    }
    deployment_url= "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/dc4baab3a2f39a75"
    endpoint = f"{deployment_url}/chat/completions?api-version=2023-05-15" # endpoint implemented in serving engine
    #print(endpoint)
    headers = {"Authorization": f"Bearer {auth_token}",
               'ai-resource-group': resource_group.value,
               "Content-Type": "application/json"}
    response = requests.post(endpoint, headers=headers, json=test_input)
    #converting the response to Json dictionary format in python to read the content
    data = response.json()
    return data['choices'][0]['message']['content']

def get_embedding(text):
    test_input = {
        "model": "text-embedding-ada-002",
        "input": [text]
    }
    deployment_url= "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d4751df26d6b379d"
    endpoint = f"{deployment_url}/embeddings?api-version=2023-05-15"  # endpoint implemented in serving engine
    headers = {
        "Authorization": f"Bearer {auth_token}",
        'ai-resource-group': os.environ['AICORE_LLM_RESOURCE_GROUP'],
        "Content-Type": "application/json"
    }
    response = requests.post(endpoint, headers=headers, json=test_input)
    data = response.json()
    return data['data'][0]['embedding']

def get_context():
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance)
    index.add(np.array(embeddings).astype('float32'))  # Add embeddings to index

    query_embedding = get_embedding(user_text)  # Your query document embedding
    D, I = index.search(np.array([query_embedding]).astype('float32'), k=2)  # Find 2 nearest neighbors

    #Combining the 2 nearest neighbous as one context
    context= all_splits[I[0][0]].page_content+ all_splits[I[0][1]].page_content
    return context



loader = PyPDFLoader("C:\\Users\\I068117\\UT_Machine Learning\\Text-Analytics_NLP\\Sentiment Analysis\\US Flex Work FAQs.pdf")
docs= loader.load()
len(docs[1].page_content)

#USing the RAG architecture, creating the chunks with a size of 2000 & overlap of 50. It can be changed

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=50, add_start_index=True)
all_splits = text_splitter.split_documents(docs[2:])
chunked_texts = [doc.page_content for doc in all_splits]

embeddings = [get_embedding(text) for text in chunked_texts]

# dimension of the first embedding
dimension = len(embeddings[0])
import time
def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


if __name__=='__main__':
# Set the app title 
    sl.title(' Personal Chatbot App')

    with sl.chat_message("assistant"):
     sl.write("Hello! 👋 I am your chatbot assistant, how can I help you?")

    if "messages" not in sl.session_state:
        sl.session_state.messages = []

    for message in sl.session_state.messages:
        with sl.chat_message(message["role"]):
            sl.markdown(message["content"])

    if user_text := sl.chat_input("How can I support you?"):
        sl.session_state.messages.append({"role": "user", "content": user_text})
        with sl.chat_message("user"):
            sl.markdown(user_text)
        
        with sl.chat_message("assistant"):
            index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance)
            index.add(np.array(embeddings).astype('float32'))  # Add embeddings to index
            query_embedding = get_embedding(user_text)  # Your query document embedding
            D, I = index.search(np.array([query_embedding]).astype('float32'), k=2)  # Find 2 nearest neighbors
            #Combining the 2 nearest neighbous as one context
            context= all_splits[I[0][0]].page_content+ all_splits[I[0][1]].page_content
            response = RAG_response_api(context, user_text)
            sl.write_stream(response_generator(response))
            sl.session_state.messages.append({"role": "assistant", "content": response})




