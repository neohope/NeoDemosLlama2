#!/usr/bin/env python3
# -*- coding utf-8 -*-

from langchain.llms import Replicate

'''
# 聊天任务
'''

if __name__ == "__main__":
    llama2_13b_chat = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    llm = Replicate(
        model=llama2_13b_chat,
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
