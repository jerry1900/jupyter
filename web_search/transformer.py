
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings



class text_to_vectorstore:
    ''' 调用text_to_vect()方法，将输入的一段文本存储在向量数据库里，并将该向量数据库作为结果返回'''


    def __init__(self, text):
        self.text = text

    def display_info(self):
        print(f"Input text as : {self.text}")

    def text_to_vect(self):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )

        splitter_content = text_splitter.split_text(self.text)


        api_key = os.getenv("OPENAI_API_KEY")

        # persist_path = r'C:\Users\PycharmProjects\langchain'

        vectorstore = Chroma.from_texts(
            splitter_content,
            embedding=OpenAIEmbeddings(
                openai_api_key=api_key,
                base_url="https://wdapi7.61798.cn/v1"),
                # persist_directory = persist_path
        )


        return vectorstore