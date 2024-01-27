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

使用Zero-Shot Learning 和 Few-Shot Learning
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

def sentiment(llm, text):
    response = chat_completion(llm, messages=[
        user("You are a sentiment classifier. For each message, give the percentage of positive/netural/negative."),
        user("I liked it"),
        assistant("70% positive 30% neutral 0% negative"),
        user("It could be better"),
        assistant("0% positive 50% neutral 50% negative"),
        user("It's fine"),
        assistant("25% positive 50% neutral 25% negative"),
        user(text),
    ])
    return response

def print_sentiment(llm, text):
    print(f'INPUT: {text}')
    print(sentiment(llm, text))

if __name__ == "__main__":
    llm=create_llm()

    # Zero-Shot Learning
    complete_and_print(llm, "Text: This was the best movie I've ever seen! \n The sentiment of the text is: ")
    # Returns positive sentiment
    complete_and_print(llm, "Text: The director was trying too hard. \n The sentiment of the text is: ")
    # Returns negative sentiment

    # Few-Shot Learning，在prompt中先提供一些例子，会提升最终效果
    print_sentiment(llm, "I thought it was okay")
    # More likely to return a balanced mix of positive, neutral, and negative
    print_sentiment(llm, "I loved it!")
    # More likely to return 100% positive
    print_sentiment(llm, "Terrible service 0/10")
    # More likely to return 100% negative

    # 指定大模型的角色，会提升最终效果
    complete_and_print("Explain the pros and cons of using PyTorch.")
    # More likely to explain the pros and cons of PyTorch covers general areas like documentation, the PyTorch community, and mentions a steep learning curve
    complete_and_print("Your role is a machine learning expert who gives highly technical advice to senior engineers who work with complicated datasets. Explain the pros and cons of using PyTorch.")
    # Often results in more technical benefits and drawbacks that provide more technical details on how model layers

    # 通过调整提示语，限制需要的输出内容
    complete_and_print(
        "Give me the zip code for Menlo Park in JSON format with the field 'zip_code'",
    )
    # Likely returns the JSON and also "Sure! Here's the JSON..."

    # 去掉不需要的内容，LLAMA2_70B_CHAT效果会比较好
    complete_and_print(
        """
        You are a robot that only outputs JSON.
        You reply in JSON format with the field 'zip_code'.
        Example question: What is the zip code of the Empire State Building? Example answer: {'zip_code': 10118}
        Now here is my question: What is the zip code of Menlo Park?
        """
    )
    # "{'zip_code': 94025}"
