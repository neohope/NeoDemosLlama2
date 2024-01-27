#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
from langchain.llms import Replicate

'''
# 聊天任务
# 获取授权： https://replicate.com/account/api-tokens
'''

LLAMA2_70B_CHAT = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"
LLAMA2_13B_CHAT = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
os.environ["REPLICATE_API_TOKEN"] = "YOUR_KEY_HERE"

if __name__ == "__main__":
    llm = Replicate(
        model=LLAMA2_13B_CHAT,
        model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
    )

    question = 'where is the capital of USA'
    answer = llm(question)
    print(question)
    print(answer)

    question = "who wrote the book Innovator's dilemma?"
    answer = llm(question)
    print(question)
    print(answer)
