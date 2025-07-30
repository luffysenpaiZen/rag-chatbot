import os
import asyncio
import sys
import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS,Chroma
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

st.title('Pdf chat with rag')
uploaded_file=st.file_uploader('upload pdf ',type='pdf')
question=st.text_input('ask a question on document')

# model
# llm = HuggingFaceEndpoint(
#         model='mistralai/Mistral-7B-Instruct-v0.1',
#         task='text-generation',
#         max_new_tokens=512,
#         temperature=0.1,)
# model=ChatHuggingFace(llm=llm)

model = ChatGroq(
        model="llama3-8b-8192",
        temperature=0,
        
    )

# model=ChatGoogleGenerativeAI(model='gemini-2.5-pro')
parser=StrOutputParser()

if uploaded_file:
    
    with open('temp.pdf','wb') as f:
        f.write(uploaded_file.read())
    
    # data loader
    loader=PyMuPDFLoader(file_path='temp.pdf')
    docs=loader.load()
    
    # spliiting
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    
    splitted_docs=splitter.split_documents(docs)
    # embeddings
    
    if sys.platform.startswith('win') and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Ensure event loop exists in this thread
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    # store
    store=FAISS.from_documents(
        embedding=embeddings,
        documents=splitted_docs
    )
    
    # retriever
    base_retriever=store.as_retriever(search_type='mmr',search_kwargs={'k':10,'fetch_k': 10})
    # compressor
    compressor=LLMChainExtractor.from_llm(llm=model)
    
    retriever=ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )
    
    prompt = PromptTemplate(
    template=(
        "You are a reliable assistant. explain answer the question in a informative way and give the page no it is related to ,Use only the information from the context below to answer the question.\n\n"
        "If the context does not contain enough information to answer, say:\n"
        "'I do not know based on the provided context.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["context", "question"]
    )
    
    # chain
    parallel_chain=RunnableParallel({
        'context':retriever,
        'question':RunnablePassthrough()
    })
    
    chain=parallel_chain | prompt | model | parser
    
    if question:
        responce=chain.invoke(question)
        st.write("### Answer:")
        st.write(responce)