#!/usr/bin/env python3
# -*- coding utf-8 -*-

import json
import requests
import langchain
from langchain.llms import Replicate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index import download_loader

'''
使用搜索引擎，对答案进行优化
pip install llama-index langchain
'''

if __name__ == "__main__":
    langchain.debug = True

    # 模型
    llama2_13b_chat = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    llm = Replicate(
        model=llama2_13b_chat,
        model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
    )

    # 可以替换为其他搜索引擎
    query = "Meta Connect"
    headers = {"X-API-Key": "YOUCOM_API_KEY"}
    data = requests.get(
        f"https://api.ydc-index.io/search?query={query}",
        headers=headers,
    ).json()
    # print(json.dumps(data, indent=2))

    # 查询结果转换为documents
    JsonDataReader = download_loader("JsonDataReader")
    loader = JsonDataReader()
    documents = loader.load_data([hit["snippets"] for hit in data["hits"]])

    # documents导入到VectorStoreIndex
    # use HuggingFace embeddings 
    embeddings = LangchainEmbedding(HuggingFaceEmbeddings())
    # create a ServiceContext instance to use Llama2 and custom embeddings
    service_context = ServiceContext.from_defaults(llm=llm, chunk_size=800, chunk_overlap=20, embed_model=embeddings)
    # create vector store index from the documents created above
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # create query engine from the index
    query_engine = index.as_query_engine(streaming=True)

    # 对相关内容提问
    query_engine.query("give me a summary").print_response_stream()
    query_engine.query("what products were announced").print_response_stream()
    query_engine.query("tell me more about Meta AI assistant").print_response_stream()
    query_engine.query("what are Generative AI stickers").print_response_stream()
