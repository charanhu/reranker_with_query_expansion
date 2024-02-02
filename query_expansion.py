# query_expansion.py
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from model import llm_1  # Assuming a generic language model for demonstration

def generate_question_response(input_text):
    # Prompt template for the language model
    prompt_template = f"""<s>[INST] <<SYS>>
You are a helpful expert assistant. Your users are asking questions about a given topic. 
Suggest up to five additional related questions to help them find the information they need for the provided question. 
Suggest only short questions without compound sentences. Provide a variety of questions that cover different aspects of the topic. 
Make sure they are complete questions and related to the original question. 

Output one question per line.
<</SYS>>

[question]: {input_text} [/INST]"""

    # Creating a prompt template for the Chat system
    template_messages = [
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    
    prompt_template = ChatPromptTemplate.from_messages([SystemMessage(content=prompt_template)] + template_messages)

    # Setting up memory and language model chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = LLMChain(llm=llm_1, prompt=prompt_template, memory=memory)

    # Invoking the language model to generate a response
    response = chain.invoke(input={"text": input_text})['text']
    response_list = [input_text] + response.split('\n')

    # remove empty strings from the list
    response_list = list(filter(None, response_list))

    return response_list
