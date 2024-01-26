#!/usr/bin/env python3
# -*- coding utf-8 -*-

from langchain.chains import ConversationChain, Replicate, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

'''
通过BufferWindow记录前面几轮的对话，提升对话效果
'''

if __name__ == "__main__":
    llama2_13b_chat = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    llm = Replicate(
        model=llama2_13b_chat,
        model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
    )

    # 保持3轮对话
    memory = ConversationBufferWindowMemory(llm=llm, k=3, memory_key="chat_history", ai_prefix="Assistant", human_prefix="User")

    # 模板
    INPUT_TEMPLATE = """你是一个中国厨师，用中文回答做菜的问题。你的回答需要满足以下要求:
    1. 你的回答必须是中文
    2. 回答限制在100个字以内

    {chat_history}

    Human: {human_input}
    Chatbot:"""

    conversation_prompt_template = PromptTemplate(
        input_variables=["chat_history", "human_input"], template=INPUT_TEMPLATE
    )

    conversation_chain_with_memory = ConversationChain(
        llm = llm,
        prompt = conversation_prompt_template,
        verbose = True,
        memory = memory,
    )

    conversation_chain_with_memory.predict(human_input="你是谁？")
    conversation_chain_with_memory.predict(human_input="鱼香肉丝怎么做？")
    conversation_chain_with_memory.predict(human_input="那宫保鸡丁呢？")
    conversation_chain_with_memory.predict(human_input="我问你的第一句话是什么？")
    # 此时就会丢失第一个问题了
    conversation_chain_with_memory.predict(human_input="我问你的第一句话是什么？")

    # 查看历史问题记录
    conversation_chain_with_memory.load_memory_variables({})
