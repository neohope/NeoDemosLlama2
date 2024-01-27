#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
from IPython.display import Markdown, display
from langchain.llms import Replicate
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

'''
# 使用Langchain和RAG
# 获取授权： https://replicate.com/account/api-tokens
'''

LLAMA2_70B_CHAT = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"
LLAMA2_13B_CHAT = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
os.environ["REPLICATE_API_TOKEN"] = "YOUR_KEY_HERE"

def md(t):
  display(Markdown(t))

if __name__ == "__main__":
    # langchain setup
    # Use the Llama 2 model hosted on Replicate
    # Temperature: Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value
    # top_p: When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens
    # max_new_tokens: Maximum number of tokens to generate. A word is generally 2-3 tokens
    llama_model = Replicate(
        model=LLAMA2_13B_CHAT,
        model_kwargs={"temperature": 0.75,"top_p": 1, "max_new_tokens":1000}
    )

    # Step 1: load the external data source. In our case, we will load Meta’s “Responsible Use Guide” pdf document.
    loader = OnlinePDFLoader("https://ai.meta.com/static-resource/responsible-use-guide/")
    documents = loader.load()

    # Step 2: Get text splits from document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    # Step 3: Use the embedding model
    model_name = "sentence-transformers/all-mpnet-base-v2" # embedding model
    model_kwargs = {"device": "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # Step 4: Use vector store to store embeddings
    vectorstore = FAISS.from_documents(all_splits, embeddings)

    # Query against your own data
    chain = ConversationalRetrievalChain.from_llm(llama_model, vectorstore.as_retriever(), return_source_documents=True)
    chat_history = []
    query = "How is Meta approaching open science in two short sentences?"
    result = chain({"question": query, "chat_history": chat_history})
    md(result['answer'])
    
    # This time your previous question and answer will be included as a chat history which will enable the ability
    # to ask follow up questions.
    chat_history = [(query, result["answer"])]
    query = "How is it benefiting the world?"
    result = chain({"question": query, "chat_history": chat_history})
    md(result['answer'])
