#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
from IPython.display import Markdown, display
from langchain.llms import Replicate
from langchain.chains import ConversationalRetrievalChain

'''
# 对话任务
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

