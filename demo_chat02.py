#!/usr/bin/env python3
# -*- coding utf-8 -*-

import os
import replicate
from IPython.display import Markdown, display

'''
# 对话任务
# 获取授权： https://replicate.com/account/api-tokens
'''

LLAMA2_70B_CHAT = "meta/llama-2-70b-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48"
LLAMA2_13B_CHAT = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
os.environ["REPLICATE_API_TOKEN"] = "YOUR_KEY_HERE"

def md(t):
  display(Markdown(t))

# text completion with input prompt
def Completion(prompt):
  output = replicate.run(
      LLAMA2_13B_CHAT,
      input={"prompt": prompt, "max_new_tokens":1000}
  )
  return "".join(output)

# chat completion with input prompt and system prompt
def ChatCompletion(prompt, system_prompt=None):
  output = replicate.run(
    LLAMA2_13B_CHAT,
    input={"system_prompt": system_prompt,
            "prompt": prompt,
            "max_new_tokens":1000}
  )
  return "".join(output)

if __name__ == "__main__":
    # Basic completion
    output = Completion(prompt="The typical color of a llama is: ")
    md(output)

    # System prompts
    output = ChatCompletion(
        prompt="The typical color of a llama is: ",
        system_prompt="respond with only one word"
    )
    md(output)

     # Response formats
    output = ChatCompletion(
        prompt="The typical color of a llama is: ",
        system_prompt="response in json format"
    )
    md(output)

    # Chats
    # example of single turn chat
    prompt_chat = "What is the average lifespan of a Llama?"
    output = ChatCompletion(prompt=prompt_chat, system_prompt="answer the last question in few words")
    md(output)

    # example without previous context. LLM's are stateless and cannot understand "they" without previous context
    prompt_chat = "What animal family are they?"
    output = ChatCompletion(prompt=prompt_chat, system_prompt="answer the last question in few words")
    md(output)

    # example of multi-turn chat, with storing previous context
    prompt_chat = """
        User: What is the average lifespan of a Llama?
        Assistant: Sure! The average lifespan of a llama is around 20-30 years.
        User: What animal family are they?
        """
    output = ChatCompletion(prompt=prompt_chat, system_prompt="answer the last question")
    md(output)

    # Zero-shot example. To get positive/negative/neutral sentiment, we need to give examples in the prompt
    prompt = '''
        Classify: I saw a Gecko.
        Sentiment: ?
        '''
    output = ChatCompletion(prompt, system_prompt="one word response")
    md(output)

    # By giving examples to Llama, it understands the expected output format.
    prompt = '''
        Classify: I love Llamas!
        Sentiment: Positive
        Classify: I dont like Snakes.
        Sentiment: Negative
        Classify: I saw a Gecko.
        Sentiment:'''
    output = ChatCompletion(prompt, system_prompt="One word response")
    md(output)

    # another zero-shot learning
    prompt = '''
        QUESTION: Vicuna?
        ANSWER:'''
    output = ChatCompletion(prompt, system_prompt="one word response")
    md(output)

    # Another few-shot learning example with formatted prompt.
    prompt = '''
        QUESTION: Llama?
        ANSWER: Yes
        QUESTION: Alpaca?
        ANSWER: Yes
        QUESTION: Rabbit?
        ANSWER: No
        QUESTION: Vicuna?
        ANSWER:'''
    output = ChatCompletion(prompt, system_prompt="one word response")
    md(output)

    # Chain-Of-Thought
    # Standard prompting
    prompt = '''
        Llama started with 5 tennis balls. It buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does Llama have now?
        '''
    output = ChatCompletion(prompt, system_prompt="provide short answer")
    md(output)

    # Chain-Of-Thought prompting
    prompt = '''
        Llama started with 5 tennis balls. It buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does Llama have now?
        Let's think step by step.
        '''
    output = ChatCompletion(prompt, system_prompt="provide short answer")
    md(output)
