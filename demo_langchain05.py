#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
import langchain
from langchain.llms import Replicate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

'''
# 对视频文本进行摘要
pip install langchain youtube-transcript-api tiktoken pytube
# 获取授权： https://replicate.com/account/api-tokens
'''

LLAMA2_70B_CHAT = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"
LLAMA2_13B_CHAT = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
os.environ["REPLICATE_API_TOKEN"] = "YOUR_KEY_HERE"

if __name__ == "__main__":
    langchain.debug = True

    loader = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=1k37OcjH7BM", add_video_info=True
    )
    # load the youtube video caption into Documents
    docs = loader.load()
    # check the docs length and content
    # len(docs[0].page_content), docs[0].page_content[:300]

    # 创建模型
    llm = Replicate(
        model=LLAMA2_13B_CHAT,
        model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
    )

    # 对文本进行总结
    prompt = ChatPromptTemplate.from_template(
        "Give me a summary of the text below: {text}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    # 但文本太长了，所以智能处理前4000个字符
    # be careful of the input text length sent to LLM
    text = docs[0].page_content[:4000]
    summary = chain.run(text)
    # this is the summary of the first 4000 characters of the video content
    print(summary)

    '''
    # 如果不做截断处理，就会报错
    # try to get a summary of the whole content
    text = docs[0].page_content
    summary = chain.run(text)
    print(summary)
    '''

    # 为了支持整个文本，需要进行特殊处理load_summarize_chain
    '''
    # 直接使用load_summarize_chain还是会出错
    # see https://python.langchain.com/docs/use_cases/summarization for more info
    chain = load_summarize_chain(llm, chain_type="stuff") # other supported methods are map_reduce and refine
    chain.run(docs)
    # same RuntimeError: Your input is too long. but stuff works for shorter text with input length <= 4096 tokens

    chain = load_summarize_chain(llm, chain_type="refine")
    # still get the "RuntimeError: Your input is too long. Max input length is 4096 tokens"
    chain.run(docs)
    '''
    # 第一种方式，是将文本进行拆分，然后使用refine
    # we need to split the long input text
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)
    # check the splitted docs lengths
    # len(split_docs), len(docs), len(split_docs[0].page_content), len(docs[0].page_content)

    # 文档拆分后，就可以进行处理了，stuff方法仍会出错，要用refine方法
    # now get the summary of the whole docs - the whole youtube content
    chain = load_summarize_chain(llm, chain_type="refine")
    chain.run(split_docs)

    # 第二种方式，是使用map_reduce
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    chain.run(split_docs)
