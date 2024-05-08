from langchain_community.llms import CTransformers

def load_llm(model_file="model/vinallama-7b-chat_q5_0.gguf"):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.01
    )
    return llm