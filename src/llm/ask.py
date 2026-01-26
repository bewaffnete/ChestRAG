from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from llm.prompt_template import prompt_template


def ask_about_photo(llm,retrieved_docs, question) -> str:

    context = "REPORT\n\n".join(
        f"[Similarity: {doc.metadata['similarity']:.4f}]\n"
        f"{doc.page_content}"

        for doc in retrieved_docs
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (
            {
                "context": lambda x: context,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    answer = chain.invoke(question)
    return answer


