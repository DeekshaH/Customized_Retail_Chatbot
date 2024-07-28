import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from huggingface_hub import hf_hub_download
import re

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

@st.cache_resource
def create_chain(system_prompt):
    (repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                                  "mistral-7b-instruct-v0.1.Q4_0.gguf")

    model_path = hf_hub_download(repo_id=repo_id,
                                 filename=model_file_name,
                                 repo_type="model")

    llm = LlamaCpp(
            model_path=model_path,
            temperature=0,
            max_tokens=512,
            top_p=1,
            stop=["[INST]"],
            verbose=False,
            streaming=True,
            )

    template = """
    <s>[INST]{}[/INST]</s>

    [INST]{}[/INST]
    """.format(system_prompt, "{question}")

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = prompt | llm

    return llm_chain

def is_retail_question(question):
    retail_keywords = [
        "product", "price", "availability", "store", "discount", "sale", "inventory",
        "return", "exchange", "customer service", "purchase", "order", "shipping", "payment",
        "refund", "brand", "warranty", "guarantee", "catalog", "checkout", "delivery",
        "stock", "retail"
    ]
    pattern = re.compile(r'\b(?:' + '|'.join(retail_keywords) + r')\b', re.IGNORECASE)
    return bool(pattern.search(question))

st.set_page_config(
    page_title="Retail Chatbot!"
)

st.header("Retail Chatbot!")

system_prompt = st.text_area(
    label="System Prompt",
    value="How may I help you?.",
    key="system_prompt")

llm_chain = create_chain(system_prompt)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Your message here", key="user_input"):

    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("user"):
        st.markdown(user_prompt)

    if is_retail_question(user_prompt):
        response = llm_chain.invoke({"question": user_prompt})
    else:
        response = "This is not my topic"

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    with st.chat_message("assistant"):
        st.markdown(response)