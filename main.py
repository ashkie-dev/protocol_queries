import glob
from langchain.chat_models import ChatOpenAI
import langchain
from llama_index import SimpleDirectoryReader, SummaryIndex, download_loader, PromptHelper, LLMPredictor, VectorStoreIndex, SummaryIndex
from llama_index.llms import OpenAI
import os
from dotenv import loadenv

os.environ['OPENAI_API_KEY'] = os.environ["OPENAI_API_TOKEN"]
# os.environ["OPENAI_API_TOKEN"]


def build_index(file_path):
    max_input_size = 4096
    num_outputs = 512
    # max_chunk_overlap = 20.00
    max_chunk_overlap = 0.50
    chunk_size_limit = 256

    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(
        temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    download_loader('SimpleDirectoryReader')
    if file_path is isinstance(file_path, list):
        documents = SimpleDirectoryReader(input_dir=file_path).load_data()
    elif isinstance(file_path, str):
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    # index = VectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = VectorStoreIndex.from_documents(documents).as_query_engine()
    return index


index = build_index(file_path=file_path)


def chatbot(prompt):
    # return index.query(prompt, response_mode="compact")
    return index.query(prompt)


while True:
    print('########################################')
    pt = input('Question: ')
    if pt.lower() == 'end':
        break
    response = chatbot(pt)
    print('Question:', pt)
    print('----------------------------------------')
    print('Answer: ')
    print(response)
