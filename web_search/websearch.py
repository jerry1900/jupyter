
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from Loader_Transformer import Loader_Transformer
from transformer import text_to_vectorstore



if __name__ == '__main__':

    loader = Loader_Transformer("https://www.baidu.com")
    text = loader.url_to_text()
    # print(text)


    vector = text_to_vectorstore(text)
    vectorstore = vector.text_to_vect()

    query = '韩国什么视频爆红？'
    search_result = vectorstore.similarity_search_with_score(query)
    print(search_result[0])

    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context and use chinese to answer:
    {context}

    Question: {question}
    """

    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        base_url="https://wdapi7.61798.cn/v1"
    )

    chain = setup_and_retrieval | prompt | llm | output_parser

    result = chain.invoke(query)
    print(result)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
