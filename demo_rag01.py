#!/usr/bin/env python3
# -*- coding utf-8 -*-

import langchain
from langchain.chains import Replicate, RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

'''
通过Augmented Generation (RAG)技术，向量存到FAISS，提升对话效果
'''

DATA_PATH = 'data_path'
DB_FAISS_PATH = 'vectorstore/db_faiss'

def load_data_to_faiss():
    # 加载PDF文档
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    #print(len(documents), documents[0].page_content[0:100])

    # 按长度截断
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(documents)
    # print(len(splits), splits[0])

    # 储存到faiss中
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(splits, embeddings)
    db.save_local(DB_FAISS_PATH)


if __name__ == "__main__":
    # 仅第一次运行
    # load_data_to_faiss()

    # 加载本地数据
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                        model_kwargs={'device': 'cpu'})
    rag_store = FAISS.load_local(DB_FAISS_PATH, embeddings)

    langchain.debug=True 
    llama2_13b_chat = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    llm = Replicate(
        model=llama2_13b_chat,
        model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
    )

    template = """
    [INST]Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
    {context}
    Question: {question} [/INST]
    """

    retriever = rag_store.as_retriever(
            search_kwargs={"k": 6}
        )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever,     
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=template,
                input_variables=["context", "question"],
            ),
        }
    )

    result = qa_chain({"query": "Why choose Llama?"})
    print(result)
