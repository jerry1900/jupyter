import re
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer

class Loader_Transformer:
    '''该类的url_to_text方法输入一个url，并将该页面的全部中文字符获取下来，拼接后返回一个大的字符串作为函数的输出 '''

    def __init__(self, url):
        self.url = url

    def display_info(self):
        print(f"URL: {self.url}")


    def url_to_text(self):

        loader = AsyncChromiumLoader([self.url])
        html = loader.load()

        print(html)

        bs_transformer = BeautifulSoupTransformer()
        docs_transformed = bs_transformer.transform_documents(
            html,
            tags_to_extract=["span", "li", "p", "div", "a"]
        )

        # print(type(docs_transformed))
        # object_size = sys.getsizeof(docs_transformed)
        # print(f"Object Size: {object_size} bytes")

        chinese_text = re.findall("[\u4e00-\u9fa5]+", docs_transformed[0].page_content)

        print(type(chinese_text), chinese_text)

        text = ''

        for i in chinese_text:
            text = text + ',' + str(i)

        return text