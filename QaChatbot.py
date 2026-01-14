from langchain_groq import ChatGroq
import os
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

def getLlm():
    os.environ["GROQ_API_KEY"] = ""
    llm = ChatGroq(
        model="llama-3.2-11b-text-preview",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
        # other params...
    )
    return llm

def loadCsv():
    loader = CSVLoader(file_path='Qa_faqs.csv', source_column="prompt")
    data = loader.load()
    return data

def getEmbedings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

def getVectorDb(data,embeddings,isReset):    
    QaVectorStore = Chroma(collection_name="Qa_description")    
    if not isReset:
        retriever = QaVectorStore.as_retriever(score_threshold = 0.7)
        return retriever
    
    QaVectorStore.delete_collection()
    QaVectorStore = Chroma.from_documents(data,
                                embedding=embeddings,
                                persist_directory='./chromadb',
                                collection_name="Qa_description")
    #QaVectorStore.persist()
    retriever = QaVectorStore.as_retriever(score_threshold = 0.7)
    return retriever

def getPromptTemplate():
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}

No pre-amble.
"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    return chain_type_kwargs

def create_qa_chain(llm,retriever,chain_type_kwargs):
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            #memory=memory,
                            return_source_documents=False,
                            chain_type_kwargs=chain_type_kwargs)
        
    return chain


user_sessions = {}
llm=None
retriever=None
chain_type_kwargs=None
def answer_question(session_id, query):
    if session_id not in user_sessions:
        user_sessions[session_id] = create_qa_chain(llm,retriever,chain_type_kwargs)
    qa_chain = user_sessions[session_id]
    result = qa_chain({"query": query})
    return result["result"]



def initializeQaChatBot():
    global llm
    global retriever
    global chain_type_kwargs
    llm=getLlm()
    data=loadCsv()
    resetDb=True
    embeddings=None
    if resetDb:
        embeddings=getEmbedings()
    retriever=getVectorDb(data,embeddings,resetDb)
    chain_type_kwargs=getPromptTemplate()
   
if __name__ == "__main__":
    initializeQaChatBot()
    ans=answer_question("12", "What is Tw?")
    print(ans)
