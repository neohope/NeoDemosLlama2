#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
import re
from statistics import mode
from typing import Dict, List
from langchain.llms import Replicate
from langchain.memory import ChatMessageHistory
from langchain.schema.messages import get_buffer_string

'''
# 使用搜索引擎，对答案进行优化
pip install llama-index langchain
# 获取授权： https://replicate.com/account/api-tokens

使用Chain-of-Thought、Self-Consistency 和 RAG
'''

LLAMA2_70B_CHAT = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"
LLAMA2_13B_CHAT = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
os.environ["REPLICATE_API_TOKEN"] = "YOUR_KEY_HERE"

def create_llm(
    model: str = LLAMA2_70B_CHAT,
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

def gen_answer():
    response = llm(
        "John found that the average of 15 numbers is 40."
        "If 10 is added to each number then the mean of the numbers is?"
        "Report the answer surrounded by three backticks, for example: ```123```"
    )
    match = re.search(r'```(\d+)```', response)
    if match is None:
        return None
    return match.group(1)

def prompt_with_rag(llm, retrived_info, question):
    complete_and_print(
        llm, 
        f"Given the following information: '{retrived_info}', respond to: '{question}'"
    )

def ask_for_temperature(llm, day):
    temp_on_day = MENLO_PARK_TEMPS.get(day) or "unknown temperature"
    prompt_with_rag(
        llm, 
        f"The temperature in Menlo Park was {temp_on_day} on {day}'",  # Retrieved fact
        f"What is the temperature in Menlo Park on {day}?",  # User question
    )

if __name__ == "__main__":
    llm=create_llm()

    # 使用思维链，会提升最终的效果
    complete_and_print(llm, "Who lived longer Elvis Presley or Mozart?")
    # Often gives incorrect answer of "Mozart"
    complete_and_print(llm, "Who lived longer Elvis Presley or Mozart? Let's think through this carefully, step by step.")
    # Gives the correct answer "Elvis"

    # 使用自洽性Self-Consistency，会提升最终的效果
    answers = [gen_answer() for i in range(5)]
    print(
        f"Answers: {answers}\n",
        f"Final answer: {mode(answers)}",
        )
    # Sample runs of Llama-2-70B (all correct):
    # [50, 50, 750, 50, 50]  -> 50
    # [130, 10, 750, 50, 50] -> 50
    # [50, None, 10, 50, 50] -> 50

    # 使用RAG，解决大模型的局限性
    complete_and_print("What is the capital of the California?", model = LLAMA2_70B_CHAT)
    # Gives the correct answer "Sacramento"
    complete_and_print("What was the temperature in Menlo Park on December 12th, 2023?")
    # "I'm just an AI, I don't have access to real-time weather data or historical weather records."
    complete_and_print("What time is my dinner reservation on Saturday and what should I wear?")
    # "I'm not able to access your personal information [..] I can provide some general guidance"

    # 通过RAG补充信息后，就可以有正确的回答
    MENLO_PARK_TEMPS = {
        "2023-12-11": "52 degrees Fahrenheit",
        "2023-12-12": "51 degrees Fahrenheit",
        "2023-12-13": "51 degrees Fahrenheit",
    }
    ask_for_temperature("2023-12-12")
    # "Sure! The temperature in Menlo Park on 2023-12-12 was 51 degrees Fahrenheit."
    ask_for_temperature("2023-07-18")
    # "I'm not able to provide the temperature in Menlo Park on 2023-07-18 as the information provided states that the temperature was unk

