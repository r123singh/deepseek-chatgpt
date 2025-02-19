import streamlit as st 
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import re


def chat_screen():
    st.title("⚡Deep-LlamaGPT🦙")
    selectedModel = st.sidebar.selectbox("Choose model", ["llama3.2:latest","deepseek-r1:1.5b "], label_visibility="hidden")
    model = Ollama(model = selectedModel)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    def DeepSeek_invoke(user_query):
        
        prompt = PromptTemplate.from_template("""You are assistant to answer user query provided:{query}.""")
        llm_chain = LLMChain(llm=model, prompt=prompt)
        result = llm_chain.run(query=user_query)
        cleaned_response = re.sub(r'<think>.*?</think>','',result,flags=re.DOTALL) if "<think>" in result else result
        
        with st.chat_message("assistant"):st.markdown(cleaned_response.strip())
        st.session_state.messages.append(
            {
            "role": "user",
            "content": user_query
            }
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": cleaned_response.strip()
            }
        )      
    user_query = st.chat_input("Message DeepSeek GPT")

    if user_query:
        with st.chat_message("user"): st.markdown(user_query)
        DeepSeek_invoke(user_query)
              
if __name__ == "__main__":
    chat_screen()
            