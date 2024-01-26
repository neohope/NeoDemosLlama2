#!/usr/bin/env python3
# -*- coding utf-8 -*-

import json
import langchain
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from langchain.utilities import SQLDatabase
from langchain.memory import ConversationBufferMemory
from langchain_experimental.sql import SQLDatabaseChain

'''
使用SQL数据库，对答案进行优化
pip install langchain replicate langchain_experimental
'''

if __name__ == "__main__":
    langchain.debug = True

    # 模型
    llama2_13b_chat = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    llm = Replicate(
        model=llama2_13b_chat,
        model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
    )

    # sqlite数据库
    db = SQLDatabase.from_uri("sqlite:///nba_roster.db", sample_rows_in_table_info= 0)

    # 模板
    PROMPT_SUFFIX = """
    Only use the following tables:
    {table_info}

    Question: {input}"""

    # chain
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_sql=True, 
                                        prompt=PromptTemplate(input_variables=["input", "table_info"], 
                                        template=PROMPT_SUFFIX))
    
    # 这样就可以用SQL中的数据，支持单次提问了
    # 由于不支持上下文，上面的第三个问题会出错
    db_chain.run("How many unique teams are there?")
    db_chain.run("Which team is Klay Thompson in?")
    db_chain.run("What's his salary?")
    
    # 如果支持类似问题，需要支持上下文
    memory = ConversationBufferMemory()
    db_chain_memory = SQLDatabaseChain.from_llm(llm, db, memory=memory, 
                                                verbose=True, return_sql=True, 
                                                prompt=PromptTemplate(input_variables=["input", "table_info"], 
                                                template=PROMPT_SUFFIX))
    # 提出第一轮问题
    question = "Which team is Klay Thompson in"
    answer = db_chain_memory.run(question)
    print(answer)

    # 提出第二轮问题，可见支持上下文了
    memory.save_context({"input": question},
                        {"output": json.dumps(answer)})
    followup = "What's his salary"
    followup_answer = db_chain_memory.run(followup)
    print(followup_answer)
