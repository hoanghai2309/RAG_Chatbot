import re
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Define a custom output parser that extends the StrOutputParser
class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()

    # Parse the text by extracting the answer
    def parse(self, text: str) -> str:
        return self.extract_answer(text)

    # Extract the answer from the text response using a regular expression pattern
    def extract_answer(self, text_response: str, pattern: str = r"Answer:\s*(.*)") -> str:
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response



from langchain.prompts import PromptTemplate
def creat_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt
template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
    {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""
prompt = creat_prompt(template)


# Define the Offline_RAG class
class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = prompt
        self.str_parser = Str_OutputParser()



    # Get the chain of operations
    def get_chain(self, retriever):
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }
        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.str_parser
        )
        return rag_chain

    # Format the documents by joining them with newlines
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
