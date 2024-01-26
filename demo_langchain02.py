#!/usr/bin/env python3
# -*- coding utf-8 -*-

from langchain.chains import LLMChain, Replicate
from langchain.prompts import PromptTemplate

'''
# 翻译任务
'''

if __name__ == "__main__":
    llama2_13b_chat = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    llm = Replicate(
        model=llama2_13b_chat,
        model_kwargs={"temperature": 0.01, "top_p": 1, "max_new_tokens":500}
    )

    template = """
    You are a Translator. Translate the following content from {input_language} to {output_language} and reply with only the translated result.
    {input_content}
    """

    translator_chain = LLMChain(
        llm = llm,
        prompt = PromptTemplate(
                template=template,
                input_variables=["input_language", "output_language", "input_content"],
            ),
    )

    answer = translator_chain.run(input_language="English", output_language="French", input_content="Who wrote the book Innovators dilemma?")
    print(answer)
