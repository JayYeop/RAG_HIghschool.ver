# rag_core.py (Google LLM + OpenAI Embedding 버전)

import os
import pickle
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
)
from langchain_community.vectorstores.faiss import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from config import (
    DOCS_DIR, KNOWLEDGE_BASE_DIR,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
)

load_dotenv()


def load_documents_from_directory(directory):
    all_documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        loader = None
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif filename.lower().endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_path)
        elif filename.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        if loader:
            try:
                print(f"{filename} 파일을 처리합니다...")
                all_documents.extend(loader.load())
            except Exception as e:
                print(f"'{filename}' 파일 처리 중 오류 발생: {e}")
    return all_documents


def load_models(api_provider, api_key):
    if not api_key:
        return None, None
    try:
        if api_provider == 'NVIDIA':
            llm = ChatNVIDIA(
                model="mistralai/mixtral-8x7b-instruct-v0.1",
                nvidia_api_key=api_key
            )
            embedder = NVIDIAEmbeddings(
                model="nvidia/nv-embed-v1",
                nvidia_api_key=api_key
            )

        elif api_provider == 'Google':
            # LLM은 Gemini 그대로 사용
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=api_key
            )
            # [수정] Google 임베딩 대신 OpenAI 임베딩 사용
            embedder = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )
            # 필요시 더 정밀한 모델:
            # embedder = OpenAIEmbeddings(model="text-embedding-3-large")

        else:
            return None, None

        return llm, embedder

    except Exception as e:
        print(f"모델 로딩 중 오류 발생: {e}")
        return None, None


def get_splitters():
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP
    )
    return parent_splitter, child_splitter


def create_and_save_retriever(embedder, kb_name):
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        return None

    raw_documents = load_documents_from_directory(DOCS_DIR)
    if not raw_documents:
        return None

    parent_splitter, child_splitter = get_splitters()
    vectorstore = FAISS.from_documents(raw_documents, embedder)
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(raw_documents, ids=None)

    kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
    os.makedirs(kb_path, exist_ok=True)

    retriever.vectorstore.save_local(os.path.join(kb_path, "faiss_index"))
    with open(os.path.join(kb_path, "docstore.pkl"), "wb") as f:
        pickle.dump(retriever.docstore, f)

    return retriever


def load_retriever(embedder, kb_name):
    try:
        kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
        vectorstore = FAISS.load_local(
            os.path.join(kb_path, "faiss_index"),
            embedder,
            allow_dangerous_deserialization=True
        )
        with open(os.path.join(kb_path, "docstore.pkl"), "rb") as f:
            store = pickle.load(f)

        parent_splitter, child_splitter = get_splitters()
        return ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
    except Exception as e:
        print(f"리트리버 '{kb_name}' 로딩 실패: {e}")
        return None


def update_and_save_retriever(embedder, kb_name):
    # 1. 기존 리트리버와 컴포넌트(vectorstore, docstore) 로드
    kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
    try:
        retriever = load_retriever(embedder, kb_name)
        if retriever is None: # 로드 실패 시 새로 생성
            return create_and_save_retriever(embedder, kb_name)
    except Exception:
        return create_and_save_retriever(embedder, kb_name)

    # 2. 새로 추가할 문서만 로드
    new_documents = load_documents_from_directory(DOCS_DIR)
    if not new_documents:
        print("추가할 새로운 문서가 없습니다.")
        return retriever

    print(f"'{kb_name}'에 {len(new_documents)}개의 새 문서를 추가합니다.")

    # 3. ParentDocumentRetriever에 새 문서 추가 (중요!)
    # add_documents가 내부적으로 자식 청크를 만들고 임베딩하여 vectorstore에 추가하고,
    # 부모 문서는 docstore에 저장해줍니다.
    retriever.add_documents(new_documents, ids=None)

    # 4. 변경된 vectorstore와 docstore를 다시 저장
    retriever.vectorstore.save_local(os.path.join(kb_path, "faiss_index"))
    with open(os.path.join(kb_path, "docstore.pkl"), "wb") as f:
        pickle.dump(retriever.docstore, f)

    return retriever


def create_rag_chain(llm, retriever, system_prompt):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    chain = prompt_template | llm | StrOutputParser()
    return chain


def get_contextual_response(user_input, retriever, chain):
    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])
    augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}\n"
    return chain.stream({"input": augmented_user_input})
