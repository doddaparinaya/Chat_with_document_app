import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
#from langchain.callbacks import get_openai_callback
import os

with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Prompt Engineer](https://youtube.com/@engineerprompt)')
 
load_dotenv()
def main():
    st.header("Chat with PDF 💬")
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    if pdf is not None:
        reader = PdfReader(pdf)
        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 200,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)
        embeddings = OpenAIEmbeddings()
        with open("foo.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        with open("foo.pkl", 'rb') as f:
            new_docsearch = pickle.load(f)
        docsearch = FAISS.from_texts(texts, new_docsearch)
        query = st.text_input("Ask questions about your PDF file:")
        if query:
            docs = docsearch.similarity_search(query)
            chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
            response=chain.run(input_documents=docs, question=query)
            st.write(response)
if __name__ == '__main__':
    main()