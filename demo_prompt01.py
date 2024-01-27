#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
from typing import Dict, List
from langchain.llms import Replicate
from langchain.memory import ChatMessageHistory
from langchain.schema.messages import get_buffer_string

'''
# 使用搜索引擎，对答案进行优化
pip install llama-index langchain
# 获取授权： https://replicate.com/account/api-tokens
'''

LLAMA2_70B_CHAT = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"
LLAMA2_13B_CHAT = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
os.environ["REPLICATE_API_TOKEN"] = "YOUR_KEY_HERE"

def create_llm(
    model: str = LLAMA2_13B_CHAT,
    temperature: float = 0.6,
    top_p: float = 0.9,):
    llm = Replicate(
            model=model,
            model_kwargs={"temperature": temperature,"top_p": top_p, "max_new_tokens": 1000}
        )
    return llm

def assistant(content: str):
    return { "role": "assistant", "content": content }

def user(content: str):
    return { "role": "user", "content": content }

def chat_completion(
    llm: Replicate,
    messages: List[Dict],
) -> str:
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        elif message["role"] == "assistant":
            history.add_ai_message(message["content"])
        else:
            raise Exception("Unknown role")
    return llm(
        get_buffer_string(
            history.messages,
            human_prefix="USER",
            ai_prefix="ASSISTANT",
        ),
    )

def complete_and_print(llm: Replicate, prompt: str):
    print(f'==============\n{prompt}\n==============')
    response = llm(prompt)
    print(response, end='\n\n')

if __name__ == "__main__":
    llm=create_llm()
    complete_and_print(llm, "The typical color of the sky is: ")
    complete_and_print(llm, "which model version are you?")

    complete_and_print(prompt="Describe quantum physics in one short sentence of no more than 12 words")
    # Returns a succinct explanation of quantum physics that mentions particles and states existing simultaneously.

    complete_and_print("Explain the latest advances in large language models to me.")
    # More likely to cite sources from 2017

    complete_and_print("Explain the latest advances in large language models to me. Always cite your sources. Never cite sources older than 2020.")
    # Gives more specific advances and only cites sources from 2020

    response = chat_completion(messages=[
        user("My favorite color is blue."),
        assistant("That's great to hear!"),
        user("What is my favorite color?"),
    ])
    print(response)
    # "Sure, I can help you with that! Your favorite color is blue."
