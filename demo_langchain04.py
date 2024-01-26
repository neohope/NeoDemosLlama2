#!/usr/bin/env python3
# -*- coding utf-8 -*-

from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

'''
进行对话
'''

if __name__ == "__main__":
    # for token-wise streaming so you'll see the answer gets generated token by token when Llama is answering your question
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    llm = LlamaCpp(
        model_path="path-to-llama-gguf-file",
        temperature=0.0,
        top_p=1,
        n_ctx=6000,
        callback_manager=callback_manager, 
        verbose=True,
    )
    
    # 直接对话
    question = "who wrote the book Innovator's dilemma?"
    answer = llm(question)
    print(question)
    print(answer)

    # 使用模板
    prompt = PromptTemplate.from_template(
        "who wrote {book}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run("innovator's dilemma")
    print(answer)

    # 使用模板
    prompt = PromptTemplate.from_template(
        "What is {what}?"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run("llama2")
