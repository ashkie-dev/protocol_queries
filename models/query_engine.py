import glob
from langchain.chat_models import ChatOpenAI
import langchain
from llama_index import SimpleDirectoryReader, SummaryIndex, download_loader, PromptHelper, LLMPredictor, VectorStoreIndex, SummaryIndex
from llama_index.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


class QueryEngine:

    def __init__(self):
        self.training_data = os.getenv('TRAINING_DATA')
        pass

    def build_index(self, **kwargs):
        self.max_input_size = kwargs.get('max_input_size', 4096)
        self.num_outputs = kwargs.get('num_outputs', 512)
        # max_chunk_overlap = 20.00
        self.max_chunk_overlap = kwargs.get('max_chunk_overlap', 0.50)
        self.chunk_size_limit = kwargs.get('chunk_size_limit', 256)

        prompt_helper = PromptHelper(
            self.max_input_size, self.num_outputs, self.max_chunk_overlap, chunk_size_limit=self.chunk_size_limit)

        llm_predictor = LLMPredictor(llm=ChatOpenAI(
            temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=self.num_outputs))

        download_loader('SimpleDirectoryReader')
        if self.training_data is isinstance(self.training_data, list):
            documents = SimpleDirectoryReader(
                input_dir=self.training_data).load_data()

        elif isinstance(self.training_data, str):
            documents = SimpleDirectoryReader(
                input_files=[self.training_data]).load_data()

        # index = VectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        self.index = VectorStoreIndex.from_documents(
            documents).as_query_engine()

        return self

    def chatbot(self, prompt):
        # return index.query(prompt, response_mode="compact")
        return self.index.query(prompt)
