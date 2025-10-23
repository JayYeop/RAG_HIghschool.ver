# rag_core.py
import os
import pickle
# import json
from dotenv import load_dotenv
import base64
# import fsspec
# --- LlamaIndex (멀티모달 RAG용) ---
# from llama_index.core import (
#     SimpleDirectoryReader,
#     StorageContext,
#     VectorStoreIndex,
#     Settings,
#     load_index_from_storage
# )
# from llama_index.readers.file import ImageReader, PDFReader # PDF와 이미지 리더를 명시적으로 임포트
# from llama_index.vector_stores.faiss import FaissVectorStore # FAISS 벡터 저장소 사용
# from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
# import faiss # FAISS 라이브러리 직접 임포트


from langchain_core.messages import HumanMessage,AIMessageChunk
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader,UnstructuredPDFLoader)
from langchain_community.vectorstores.faiss import FAISS #FAISS
from langchain.storage import InMemoryStore

from langchain.retrievers import ParentDocumentRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from config import DOCS_DIR, KNOWLEDGE_BASE_DIR, SYSTEM_PROMPTS,LANG_TEXT,CONTEXTUALIZE_Q_PROMPTS
from langchain_core.prompts import MessagesPlaceholder
from config import (
    DOCS_DIR, KNOWLEDGE_BASE_DIR,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
)

load_dotenv()


def load_and_process_documents(llm: ChatGoogleGenerativeAI, directory: str):
    """
    지정된 디렉토리에서 텍스트와 이미지 문서를 모두 로드합니다.
    이미지 파일은 Vision LLM을 통해 텍스트로 변환됩니다.
    """
    all_documents = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        # --- 이미지 파일 처리 ---
        if file_ext in image_extensions:
            if isinstance(llm, ChatGoogleGenerativeAI):
                print(f"🖼️  '{filename}' 이미지 파일을 분석하여 텍스트로 변환합니다...")
                description = describe_image_with_vision(llm, file_path)
                if description:
                    # 이미지 설명을 page_content로, 파일명을 source로 하는 Document 객체 생성
                    doc = Document(page_content=description, metadata={"source": filename})
                    all_documents.append(doc)
            else:
                 print(f"⚠️ '{filename}'은 이미지 파일이지만, 현재 LLM은 Vision을 지원하지 않아 건너뜁니다.")
            continue

        # --- 기존 텍스트 파일 처리 ---
        loader = None
        if file_ext == '.pdf':
            
            loader = UnstructuredPDFLoader(file_path, mode="elements",languages=['kor','eng'])
        elif file_ext == '.docx':
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_ext == '.pptx':
            loader = UnstructuredPowerPointLoader(file_path)
        elif file_ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        
        if loader:
            try:
                print(f"📄 '{filename}' 텍스트 파일을 처리합니다...")
                all_documents.extend(loader.load())
            except Exception as e:
                print(f"'{filename}' 파일 처리 중 오류 발생: {e}")
    
    return all_documents
# def load_documents_from_directory(directory):
#     all_documents = []
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         loader = None
#         if filename.lower().endswith('.pdf'):
#             loader = PyPDFLoader(file_path)
#         elif filename.lower().endswith('.docx'):
#             loader = UnstructuredWordDocumentLoader(file_path)
#         elif filename.lower().endswith('.pptx'):
#             loader = UnstructuredPowerPointLoader(file_path)
#         elif filename.lower().endswith('.txt'):
#             loader = TextLoader(file_path, encoding='utf-8')
#         if loader:
#             try:
#                 print(f"{filename} 파일을 처리합니다...")
#                 all_documents.extend(loader.load())
#             except Exception as e:
#                 print(f"'{filename}' 파일 처리 중 오류 발생: {e}")
#     return all_documents


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
                google_api_key=api_key,
                timeout=120.0
            )
            
            embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)
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


def create_and_save_retriever(llm,embedder, kb_name):
    if not os.path.exists(DOCS_DIR) or not os.listdir(DOCS_DIR):
        return None

    raw_documents = load_and_process_documents(llm,DOCS_DIR)
    if not raw_documents:
        return None
    # raw_documents 리스트에서 page_content가 비어있지 않은(non-empty)
    # Document 객체들만 남기고 리스트를 새로 만듭니다.
    print(f"필터링 전 문서 개수: {len(raw_documents)}")
    filtered_documents = [doc for doc in raw_documents if doc.page_content.strip()]
    print(f"필터링 후 문서 개수: {len(filtered_documents)}")
    
    # 만약 모든 문서가 필터링되어 아무것도 남지 않았다면, 종료합니다.
    if not filtered_documents:
        print("⚠️ 내용이 있는 유효한 문서가 없습니다.")
        return None
    
    parent_splitter, child_splitter = get_splitters()
    vectorstore = FAISS.from_documents(filtered_documents, embedder)
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        # search_kwargs={'k': 3} # 최종적으로 반환할 부모 문서의 개수를 3개로 지정
    )
    # retriever.vectorstore.search_type = "mmr"
    # retriever.vectorstore.search_kwargs.update({'fetch_k': 10}) # MMR 계산을 위해 먼저 가져올 자식 문서 개수
    retriever.add_documents(filtered_documents, ids=None)

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


def update_and_save_retriever(llm,embedder, kb_name):
    # 1. 기존 리트리버와 컴포넌트(vectorstore, docstore) 로드
    kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
    try:
        retriever = load_retriever(embedder, kb_name)
        if retriever is None: # 로드 실패 시 새로 생성
            return create_and_save_retriever(llm,embedder, kb_name)
    except Exception:
        return create_and_save_retriever(llm,embedder, kb_name)

    # 2. 새로 추가할 문서만 로드
    new_documents = load_and_process_documents(llm,DOCS_DIR)
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
# --- 🚨 새로운 대화형 RAG 체인 생성 함수 추가 ---

def create_conversational_rag_chain(llm, retriever, system_prompt, contextualize_q_system_prompt):
    """
    대화 기록을 인지하는 RAG 체인을 생성합니다.
    """
    # 1. 컨텍스트화 프롬프트 (Contextualize Prompt)
    #    🚨 하드코딩된 문자열 대신, 인자로 받은 프롬프트를 사용합니다.
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    # 2. 히스토리 인지 리트리버 (History-Aware Retriever) 생성
    #    이 리트리버는 위 프롬프트를 사용하여 질문을 재구성하고, 그 재구성된 질문으로 문서를 검색합니다.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. 답변 생성 프롬프트 (Answer Generation Prompt)
    #    이 프롬프트는 검색된 문서(context)를 바탕으로 최종 답변을 생성하도록 지시합니다.
    qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the user's question based on the following context and the chat history.

Context:
{context}"""), # <--- 바로 이 부분입니다! {context}를 위한 자리를 만들어주세요.
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
    # create_stuff_documents_chain은 검색된 모든 문서를 하나의 프롬프트에 '채워넣는(stuff)' 체인입니다.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 4. 검색 체인과 답변 생성 체인 결합
    #    이것이 최종적으로 사용될 대화형 RAG 체인입니다.
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


# --- 🚨 새로운 대화형 RAG 답변 생성 함수 추가 ---

def get_response(user_input, chat_history, rag_chain):
    """
    대화형 RAG 체인을 사용하여 스트리밍 답변과 소스 문서를 반환합니다.
    """
    # 체인을 스트리밍으로 호출합니다. chat_history를 함께 전달하는 것이 핵심입니다.
    response_stream = rag_chain.stream(
        {"input": user_input, "chat_history": chat_history}
    )
    
    # 🚨 스트림에서 넘어오는 데이터의 형식이 다릅니다. 'answer'와 'context' 키로 구분됩니다.
    full_response = ""
    sources = []
    
    for response in response_stream:
        if "answer" in response:
            full_response += response["answer"]
            yield {"chunk": response["answer"]}
        if "context" in response and response["context"]:
            print("--- [참고된 텍스트] LANGCHAIN RAG CONTEXT ---")
            context_text = "\n\n---\n\n".join([doc.page_content for doc in response["context"]])
            print(context_text)
            print("--- [참고된 텍스트] ---")
            # 소스 정보는 마지막에 한 번에 넘어오는 경우가 많습니다.
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in response["context"]]))
    
    # 스트림이 끝난 후, 최종적으로 수집된 소스 정보를 한 번에 전달합니다.
    yield {"sources": sources}

def get_contextual_response(user_input, retriever, chain):
    # --- 🚨 수정: 소스 정보를 추출하고 함께 yield 하도록 변경 ---
    docs = retriever.invoke(user_input)
    
    # 소스 정보 추출 (중복 제거 포함)
    sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
    sources.sort()

    # 첫 번째 yield: 검색된 소스 정보를 먼저 전달
    yield {"sources": sources}

    context = "\n\n".join([doc.page_content for doc in docs])
    augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}\n"
    
    # 두 번째 yield: 기존처럼 답변 스트림을 전달 (이제는 딕셔너리로 감싸서)
    stream_iterator = chain.stream({"input": augmented_user_input})
    for chunk in stream_iterator:
        yield {"chunk": chunk}

# rag_core.py 파일 맨 아래에 추가

def image_to_base64(image_file):
    """Streamlit의 UploadedFile 객체(이미지)를 base64 문자열로 변환합니다."""
    image_file.seek(0)
    image_bytes = image_file.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    return image_b64

# --- Vision 함수 1: 파일 업로드(Base64) 방식 ---
def get_response_with_vision_from_file(llm: ChatGoogleGenerativeAI, image_file, question: str, system_prompt: str):
    """RAG 검색 없이, 업로드된 이미지 파일을 직접 보고 질문에 답변하는 Gemini Vision 함수입니다."""
    if not isinstance(llm, ChatGoogleGenerativeAI):
        warning_text = "Warning: Image analysis is only supported by Google (Gemini) models."
        yield AIMessageChunk(content=warning_text)
        return
    try:
        image_b64 = image_to_base64(image_file)
        image_b64 = image_b64.strip().replace('\n', '').replace('\r', '') #Rectification
        image_mime_type = image_file.type
        image_data_url = f"data:{image_mime_type};base64,{image_b64}"

        message = HumanMessage(
            content=[
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": f"Question: {question}"},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        )

        # --- 🚨 LLM 스트림 디버깅 로직 적용 (여기가 return llm.stream(...)을 대체합니다) ---
        print(f"--- [DEBUG_LLM_STREAM] Requesting LLM stream for model: {llm.model} (File Upload) ---")
        stream_iterator = llm.stream([message]) # LLM 스트림 객체를 받음
        
        first_chunk_received = False
        full_llm_debug_response = ""

        for i, chunk in enumerate(stream_iterator): 
            if not first_chunk_received:
                print(f"--- [DEBUG_LLM_STREAM] First chunk received! (Index: {i}) ---")
                first_chunk_received = True
            
            # chunk 객체의 실제 타입과 내용 확인
            print(f"--- [DEBUG_LLM_STREAM] Chunk {i} Type: {type(chunk)}, Content (first 50 chars): {chunk.content[:50] if hasattr(chunk, 'content') else 'N/A'} ---")
            
            full_llm_debug_response += chunk.content if hasattr(chunk, 'content') else ""
            yield chunk # 원래 하던 대로 Streamlit으로 청크를 넘겨줍니다.

        if not first_chunk_received:
            print("--- [DEBUG_LLM_STREAM] No chunks received from LLM stream. ---")
        else:
            print(f"--- [DEBUG_LLM_STREAM] Full LLM response: \n{full_llm_debug_response} ---")
        # --- 🚨 LLM 스트림 디버깅 로직 끝 ---

    except Exception as e:
        error_text = f"Error processing uploaded image: {e}"
        yield AIMessageChunk(content=error_text)
        return


# --- Vision 함수 2: 공개 URL 방식 ---
def get_response_with_vision_from_url(llm: ChatGoogleGenerativeAI, image_url: str, question: str, system_prompt: str):
    """RAG 검색 없이, 공개 URL을 통해 이미지를 직접 분석하는 Gemini Vision 함수입니다."""
    if not isinstance(llm, ChatGoogleGenerativeAI):
        warning_text = "Warning: Image analysis is only supported by Google (Gemini) models. Please select Google as the AI provider in the sidebar."
        yield AIMessageChunk(content=warning_text)
        return
    
    if not image_url or not image_url.strip().startswith(("http://", "https://")):
        error_text = "Error: Please provide a valid URL starting with http:// or https://."
        yield AIMessageChunk(content=error_text)
        return

    message = HumanMessage(
        content=[
            {"type": "text", "text": system_prompt},
            {"type": "text", "text": f"Question: {question}"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    )
    
    # --- 🚨 LLM 스트림 디버깅 로직 적용 ---
    print(f"--- [DEBUG_LLM_STREAM] Requesting LLM stream for model: {llm.model} (URL) ---")
    stream_iterator = llm.stream([message])
    
    first_chunk_received = False
    full_llm_debug_response = ""

    for i, chunk in enumerate(stream_iterator): 
        if not first_chunk_received:
            print(f"--- [DEBUG_LLM_STREAM] First chunk received! (Index: {i}) ---")
            first_chunk_received = True
        
        print(f"--- [DEBUG_LLM_STREAM] Chunk {i} Type: {type(chunk)}, Content (first 50 chars): {chunk.content[:50] if hasattr(chunk, 'content') else 'N/A'} ---")
        
        full_llm_debug_response += chunk.content if hasattr(chunk, 'content') else ""
        yield chunk

    if not first_chunk_received:
        print("--- [DEBUG_LLM_STREAM] No chunks received from LLM stream. ---")
    else:
        print(f"--- [DEBUG_LLM_STREAM] Full LLM response: \n{full_llm_debug_response} ---")
    # --- 🚨 LLM 스트림 디버깅 로직 끝 ---


# --- 새로운 'Vision + RAG' 융합 함수 ---
def get_fused_vision_rag_response(llm: ChatGoogleGenerativeAI, retriever, image_file, question: str, system_prompt: str):
    """
    1. Vision으로 이미지의 핵심 개념을 추출하고,
    2. 추출된 개념으로 RAG 검색을 수행한 뒤,
    3. 이미지와 검색된 문서를 종합하여 최종 답변을 생성하는 융합 파이프라인입니다.
    """
    # --- 1단계: 이미지에서 핵심 개념(키워드) 추출 ---
    concept_extraction_prompt = """Analyze the provided image and identify the single most important technical concept or topic it represents. 
    Respond with ONLY that concept phrase, in 1 to 5 words. Do not add any explanation.
    Example responses: 'Ohm's Law', 'Kirchhoff's Current Law', 'Low Pass Filter', 'Thévenin's theorem'.
    """
    
    # 개념 추출 전용으로 새 LLM 인스턴스를 만듭니다 (스트림을 사용하지 않기 위함).
    # 주의: llm.model_name 대신 llm.model을 사용하는 것이 모델 ID를 더 정확하게 반영합니다.
    concept_extractor_llm = ChatGoogleGenerativeAI(model=llm.model, google_api_key=llm.google_api_key)
    
    try:
        image_b64 = image_to_base64(image_file)
        image_mime_type = image_file.type
        image_data_url = f"data:{image_mime_type};base64,{image_b64}"

        concept_message = HumanMessage(
            content=[
                {"type": "text", "text": concept_extraction_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        )
        extracted_concept = concept_extractor_llm.invoke([concept_message]).content.strip()
        print(f"--- [Vision->RAG] 1. Extracted Concept: '{extracted_concept}' ---")

    except Exception as e:
        yield AIMessageChunk(content=f"Error during concept extraction from image: {e}")
        return

    # --- 2단계: 추출된 개념으로 RAG 문서 검색 ---
    try:
        # 🚨 검색되는 문서 수를 제한하여 토큰 오버플로우 방지 및 응답 속도 향상
        retrieved_docs = retriever.invoke(extracted_concept, k=3) 
        # 소스 정보 추출
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
        sources.sort()
        # 소스 정보를 먼저 전달
        yield {"sources": sources}
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        print(f"--- [Vision->RAG] 2. Retrieved Context Length: {len(retrieved_context)} characters ---")
        
        if not retrieved_context:
            retrieved_context = "No relevant documents found in the knowledge base."
            
    except Exception as e:
        yield AIMessageChunk(content=f"Error during document retrieval with RAG: {e}")
        return

    # --- 3단계: 이미지 + 검색된 문서를 종합하여 최종 답변 생성 ---
    final_generation_prompt = system_prompt.format(context=retrieved_context)
    final_message = HumanMessage(
        content=[
            {"type": "text", "text": final_generation_prompt},
            {"type": "text", "text": f"User Question: {question}"},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]
    )
    
    # --- 🚨 LLM 스트림 디버깅 로직 적용 (여기가 return llm.stream(...)을 대체합니다) ---
    print(f"--- [DEBUG_LLM_STREAM] Requesting LLM stream for model: {llm.model} (Fused Vision+RAG) ---")
    stream_iterator = llm.stream([final_message])
    
    first_chunk_received = False
    full_llm_debug_response = ""

    for i, chunk in enumerate(stream_iterator): 
        if not first_chunk_received:
            print(f"--- [DEBUG_LLM_STREAM] First chunk received! (Index: {i}) ---")
            first_chunk_received = True
        
        print(f"--- [DEBUG_LLM_STREAM] Chunk {i} Type: {type(chunk)}, Content (first 50 chars): {chunk.content[:50] if hasattr(chunk, 'content') else 'N/A'} ---")
        
        full_llm_debug_response += chunk.content if hasattr(chunk, 'content') else ""
        yield chunk

    if not first_chunk_received:
        print("--- [DEBUG_LLM_STREAM] No chunks received from LLM stream. ---")
    else:
        print(f"--- [DEBUG_LLM_STREAM] Full LLM response: \n{full_llm_debug_response} ---")
    # --- 🚨 LLM 스트림 디버깅 로직 끝 ---
# rag_core.py의 멀티모달 함수 3개를 아래 코드로 교체하세요.


def describe_image_with_vision(llm: ChatGoogleGenerativeAI, image_path: str) -> str:
    """지정된 이미지 파일 경로를 받아 Vision LLM을 통해 상세한 텍스트 설명을 반환합니다."""
    try:
        # 1. MIME 타입 추론 (파일 확장자 기반)
        import mimetypes
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'image/jpeg' # 기본값

        # 2. 이미지 파일을 열고 Base64로 인코딩
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode('utf-8')

        image_data_url = f"data:{mime_type};base64,{image_b64}"

        # 3. 이미지 설명을 위한 프롬프트
        prompt = """You are an expert in analyzing images. Describe the following image in detail, including all visible text, objects, scenes, and concepts. The description should be comprehensive and optimized for later text-based semantic search. Respond only with the description, without any introductory phrases."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}}
            ]
        )
        
        # 스트리밍 없이 한 번에 답변을 받음
        response = llm.invoke([message])
        print(response)
        return response.content
    except Exception as e:
        print(f"❌ 이미지 설명 생성 중 오류 발생 ({os.path.basename(image_path)}): {e}")
        return "" # 오류 발생 시 빈 문자열 반환
# rag_core.py 파일 맨 아래에 아래 함수를 추가하세요.
import json

# def stream_study_guide(llm, chat_history: list):
#     """
#     [이미지 제외 버전] 텍스트 채팅 기록만을 분석하여 Markdown 형식의 학습 노트를 스트리밍으로 생성합니다.
#     """
#     print("🧠 (텍스트 전용) 학습 노트 스트리밍 생성을 시작합니다...")

#     # 1. [수정] 이미지 정보를 제외하고 순수 텍스트 대화 기록만으로 재구성
#     history_for_prompt = []
#     for msg in chat_history:
#         role = "학생" if msg["role"] == "user" else "AI 튜터"
#         content = msg["content"]
        
#         # 이미지와 관련된 메시지는 건너뛰거나 내용에서 이미지 관련 언급을 제거할 수 있습니다.
#         # 여기서는 간단하게 텍스트 내용만 포함합니다.
#         if content: # 내용이 비어있지 않은 경우에만 추가
#             history_for_prompt.append(f'{role}: {content}')
    
#     formatted_history = "\n".join(history_for_prompt)

#     # 2. [수정] 시스템 프롬프트에서 이미지 관련 모든 지시사항 제거
#     system_prompt = f"""
# You are an expert tutor creating a study guide. Analyze the provided conversation history between a student and an AI tutor. Your task is to generate a well-structured study guide in Markdown format based on the key topics discussed in the text.

# The final output MUST strictly be in Markdown format and include the following sections:

# 1.  **# 학습 노트: [대화의 핵심 주제]**
#     - 대화의 전체 주제를 요약하여 제목을 작성해주세요.

# 2.  **## 📝 핵심 개념 요약**
#     - 대화에서 다루어진 가장 중요한 개념, 공식, 원리 등을 3~5개의 글머리 기호(bullet points)로 요약해주세요.

# 3.  **## ✍️ 복습 퀴즈 (3문제)**
#     - 대화의 핵심 내용을 바탕으로, 학생이 자신의 이해도를 점검할 수 있는 객관식 또는 단답형 문제 3개를 만들어주세요.
#     형식:
#         **[문제 1]** (질문 내용)
#         **[문제 2]** (질문 내용)
#         **[문제 3]** (질문 내용)...
#         ...
#         ----------------------
#         **[문제 1]**
#         <정답 및 해설>
#         (답변 내용)
#         **[문제 2]** 
#         <정답 및 해설>
#         (답변 내용)

# Please generate the entire study guide based on the conversation below.

# ---
# [대화 기록]
# {formatted_history}
# ---
# """
    
#     try:
#         # invoke 대신 stream을 사용합니다.
#         stream = llm.stream(system_prompt)
#         for chunk in stream:
#             yield chunk.content # content 부분만 yield
#         print("✅ 학습 노트 스트리밍 생성이 완료되었습니다.")

#     except Exception as e:
#         print(f"❌ 학습 노트 생성 중 오류 발생: {e}")
#         yield f"학습 노트를 생성하는 중 오류가 발생했습니다: {e}"

def stream_study_guide_optimized(llm, chat_history: list):
    """
    [최적화 버전] 2단계 요약 전략을 사용하여 학습 노트를 매우 빠르게 생성합니다.
    1. 대화 기록의 핵심 내용을 먼저 요약합니다.
    2. 요약된 내용을 기반으로 최종 학습 노트를 생성합니다.
    """
    print("🧠 (최적화 버전) 학습 노트 생성을 시작합니다...")

    # --- 1단계: 대화 기록의 핵심 내용 요약 ---
    # 대화 기록을 프롬프트용으로 포맷팅하는 것은 동일합니다.
    history_for_prompt = []
    for msg in chat_history:
        role = "학생" if msg["role"] == "user" else "AI 튜터"
        content = msg["content"]
        if content:
            history_for_prompt.append(f'{role}: {content}')
    formatted_history = "\n".join(history_for_prompt)

    # 요약을 위한 간결한 프롬프트
    summarization_prompt = f"""
Analyze the following conversation history between a student and an AI tutor.
Identify and list the main topics, key concepts, and important questions discussed.
Respond ONLY with a concise summary in bullet points.

---
[Conversation History]
{formatted_history}
---
"""
    try:
        print("➡️ 1단계: 대화 내용 요약을 요청합니다...")
        # 여기서는 스트리밍이 아닌, invoke를 사용해 요약본 전체를 한 번에 받습니다.
        summary_response = llm.invoke(summarization_prompt)
        conversation_summary = summary_response.content
        print(f"✅ 1단계 요약 완료:\n{conversation_summary}")

    except Exception as e:
        print(f"❌ 1단계 요약 중 오류 발생: {e}")
        yield f"학습 노트 생성을 위한 대화 요약 중 오류가 발생했습니다: {e}"
        return


    # --- 2단계: 요약본을 기반으로 최종 학습 노트 생성 ---
    # 기존의 상세한 학습 노트 생성 프롬프트를 사용하되,
    # 방대한 대화 기록 대신 '핵심 요약본'을 넣어줍니다.
    study_guide_prompt = f"""
You are an expert tutor creating a study guide.
Your task is to generate a well-structured study guide in Markdown format based on the provided 'Conversation Summary'.

The final output MUST strictly be in Markdown format and include the following sections:

1.  **# 학습 노트: [대화의 핵심 주제]**
    - 대화의 전체 주제를 요약하여 제목을 작성해주세요.

2.  **## 📝 핵심 개념 요약**
    - 대화에서 다루어진 가장 중요한 개념, 공식, 원리 등을 3~5개의 글머리 기호(bullet points)로 요약해주세요.

3.  **## ✍️ 복습 퀴즈 (3문제)**
    - 대화의 핵심 내용을 바탕으로, 학생이 자신의 이해도를 점검할 수 있는 문제 3개를 만들어주세요.

    **[퀴즈 형식 규칙]**
    - 만약 전기회로 문제가 포함된다면, **반드시 한 문제 이상을 텍스트 기반의 ASCII 아트(ASCII Art)를 사용**하여 회로를 시각적으로 표현해야 합니다.
    **단 생성된 ASCII 아트는 반드시 Markdown 코드 블록(```)으로 감싸야 합니다.**
    - **ASCII 아트 예시:**
      ```
        R1 (4Ω)            R2 (8Ω)
    +----/\/\/\/\--------+----/\/\/\/\----+
    |                                     |
    |                                     |
    +-[(+) V1: 12V (-)]+----------------|
                                          |
                                          |
                                        (GND)
      ```

    **[전체 출력 형식]**
    **[문제 1]**
    (ASCII 아트 회로도, 필요한 경우 포함하기)
    (질문 내용)


    **[문제 2]**
    (질문 내용)


    **[문제 3]**
    (질문 내용)
    ----------------------
    **[문제 1]** <정답 및 해설>
    (답변 내용)


    **[문제 2]** <정답 및 해설>
    (답변 내용)


    **[문제 3]** <정답 및 해설>
    (답변 내용)


Please generate the entire study guide based on the conversation summary below.

---
[Conversation Summary]
{conversation_summary}
---
"""
    try:
        print("➡️ 2단계: 요약본 기반으로 학습 노트 생성을 요청합니다...")
        # 이제 훨씬 가벼워진 프롬프트로 스트리밍을 호출합니다.
        stream = llm.stream(study_guide_prompt)
        for chunk in stream:
            yield chunk.content
        print("✅ 학습 노트 스트리밍 생성이 완료되었습니다.")

    except Exception as e:
        print(f"❌ 2단계 학습 노트 생성 중 오류 발생: {e}")
        yield f"학습 노트를 생성하는 중 오류가 발생했습니다: {e}"
import re
from fpdf import FPDF

class PDFWithHeaderFooter(FPDF):
    def header(self):
        self.set_font('NotoSansKR', 'B', 12)
        self.cell(0, 10, 'EE-Assistant AI 학습 노트', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('NotoSansKR', '', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# [수정됨] 더 똑똑하고 안정적인 헬퍼 함수
def write_formatted_line(pdf, line_text, font_family, default_size=11, prefix=""):
    """
    한 줄의 텍스트를 파싱하여 '**' 부분을 굵게 처리하고, 접두사(e.g., 글머리 기호)를 추가합니다.
    이 함수는 multi_cell처럼 작동하여 다음 요소에 영향을 주지 않습니다.
    """
    # 1. 현재 커서 위치를 저장합니다.
    start_x = pdf.get_x()
    start_y = pdf.get_y()

    # 2. 접두사(글머리 기호)가 있다면 먼저 출력합니다.
    if prefix:
        pdf.set_font(font_family, '', size=default_size)
        pdf.write(h=7, text=prefix)

    # 3. 텍스트의 나머지 부분을 파싱하며 출력합니다.
    parts = re.split(r'(\*\*.*?\*\*)', line_text)
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            pdf.set_font(font_family, 'B', size=default_size)
            pdf.write(h=7, text=part[2:-2])
        else:
            pdf.set_font(font_family, '', size=default_size)
            pdf.write(h=7, text=part)

    # 4. (핵심!) 출력이 끝난 후, 커서를 다음 줄 맨 앞으로 강제 이동시킵니다.
    #    이렇게 하면 다음 요소가 항상 올바른 위치에서 시작하는 것을 보장합니다.
    pdf.ln(7)


def save_markdown_to_pdf(markdown_content: str) -> bytes:
    print("📄 [fpdf2] Markdown을 PDF로 변환합니다...")
    print(markdown_content)

    font_dir = "fonts"
    regular_font_path = os.path.join(font_dir, "NotoSansKR-Regular.ttf")
    bold_font_path = os.path.join(font_dir, "NotoSansKR-Bold.ttf")
    monospace_font_path = os.path.join(font_dir, "D2Coding-Ver1.3.2.ttf")

    pdf = PDFWithHeaderFooter()
    font_family = "NotoSansKR"
    monospace_family = "D2Coding"
    
    try:
        pdf.add_font(font_family, "", regular_font_path, uni=True)
        pdf.add_font(font_family, "B", bold_font_path, uni=True)
        if os.path.exists(monospace_font_path):
            pdf.add_font(monospace_family, "", monospace_font_path, uni=True)
        else:
            print(f"⚠️ 경고: 고정폭 폰트 '{monospace_font_path}'를 찾을 수 없습니다. ASCII 아트가 깨질 수 있습니다.")
            monospace_family = "courier" # FPDF 기본 내장 고정폭 폰트로 대체
    except Exception as e:
        font_family = "helvetica"
    
    pdf.set_font(font_family, size=11)
    pdf.add_page()
    # --- ▼▼▼ ASCII 아트 감지를 위한 상태 변수 추가 ▼▼▼ ---
    is_ascii_art_block = False
    ascii_art_buffer = []
    # [수정됨] 메인 루프를 더 단순하고 명확하게 변경
    for line in markdown_content.split('\n'):
        if line.strip() == "```":
            if not is_ascii_art_block:
                is_ascii_art_block = True
            else: # 코드 블록 끝
                is_ascii_art_block = False
                if ascii_art_buffer:
                    # 버퍼에 쌓인 ASCII 아트를 고정폭 폰트로 한 번에 출력
                    pdf.set_font(monospace_family, '', size=10) # 폰트 크기를 약간 작게
                    pdf.set_fill_color(245, 245, 245) # 연한 회색 배경
                    ascii_text = "\n".join(ascii_art_buffer)
                    pdf.multi_cell(0, 5, ascii_text, border=1, ln=1, fill=True,align='C')
                    pdf.set_font(font_family, '', size=11) # 원래 폰트로 복귀
                    ascii_art_buffer = [] # 버퍼 비우기

            continue # ``` 라인은 출력하지 않음

        if is_ascii_art_block:
            ascii_art_buffer.append(line)
            continue
        # --- ▲▲▲ ASCII 아트 블록 처리 로직 끝 ▲▲▲ ---

        line = line.strip()
        if not line:
            continue
        
        if line.startswith('# '):
            pdf.set_font(font_family, 'B', size=24)
            pdf.set_text_color(40, 40, 120)
            pdf.multi_cell(0, 15, line.replace('# ', '').strip(), ln=1, align='C') # 높이 조절
            pdf.set_text_color(0, 0, 0)
            pdf.ln(10)

        elif line.startswith('## '):
            pdf.set_font(font_family, 'B', size=16)
            pdf.set_fill_color(224, 235, 255)
            # multi_cell 대신 cell을 써야 배경색이 텍스트 높이에 맞게 깔끔하게 들어갑니다.
            pdf.cell(0, 10, line.replace('## ', '').strip(), ln=1, align='C', fill=True)
            pdf.ln(5)
        
        elif line.startswith('----------------------'):
            pdf.add_page()

        elif line.startswith('* '):
            # 글머리 기호를 접두사로, 나머지 텍스트를 내용으로 헬퍼 함수에 전달
            write_formatted_line(pdf, line[2:].strip(), font_family, default_size=11, prefix="  •  ")

        else: # [문제], <정답및해설>, 일반 텍스트 모두 이 곳에서 처리
            write_formatted_line(pdf, line, font_family)

    print("✅ 'design_preview.pdf' 파일이 멋지게 생성되었습니다!")
    return bytes(pdf.output(dest='S'))













# def save_markdown_to_pdf(markdown_content: str) -> bytes:
#     """
#     [fpdf2 최종 버전 - 레이아웃 수정] Markdown 텍스트를 안정적인 PDF로 변환합니다.
#     프로젝트 내부에 포함된 한글 폰트를 사용합니다.
#     """
#     print("📄 [fpdf2] Markdown을 PDF로 변환합니다...")

#     font_dir = "fonts"
#     regular_font_path = os.path.join(font_dir, "NotoSansKR-Regular.ttf")
#     bold_font_path = os.path.join(font_dir, "NotoSansKR-Bold.ttf")

#     pdf = FPDF()
#     pdf.add_page()
#     font_family = "NotoSansKR"

#     try:
#         if not os.path.exists(regular_font_path) or not os.path.exists(bold_font_path):
#              raise FileNotFoundError("폰트 파일('NotoSansKR-Regular.ttf' 또는 'NotoSansKR-Bold.ttf')을 'fonts' 폴더에서 찾을 수 없습니다.")

#         pdf.add_font(font_family, "", regular_font_path, uni=True)
#         pdf.add_font(font_family, "B", bold_font_path, uni=True)
#         pdf.set_font(font_family, size=11)
        
#     except Exception as e:
#         print(f"⚠️ 경고: 한글 폰트 로드 중 오류 발생: {e}")
#         pdf.set_font("helvetica", size=11)
#         font_family = "helvetica"

#     # --- ▼▼▼ 핵심 수정 부분 (모든 multi_cell에 ln=1 추가) ▼▼▼ ---
#     for line in markdown_content.split('\n'):
#         line = line.strip()
#         if line.startswith('# '):  #Main-Title
#             pdf.set_font(font_family, 'B', size=24)
#             pdf.set_text_color(80,91,166)
#             pdf.multi_cell(0, 12, line.replace('# ', '').strip(), ln=1, align='C')
#             pdf.ln(5) # 제목 아래에 추가 간격
#         elif line.startswith('## '):
#             pdf.set_font(font_family, 'B', size=18)
#             # --- 디자인 추가 ---
#             pdf.set_fill_color(230, 230, 230)  # 연한 회색 배경 설정
#             pdf.set_text_color(0, 0, 0)       # 텍스트 색상은 검은색으로
#             pdf.set_draw_color(0, 80, 180)    # 선 색상을 파란색 계열로
#             pdf.set_line_width(0.5)           # 선 두께 설정
#             # 제목 텍스트를 배경색이 채워진 셀에 쓴다
#             pdf.cell(0, 10, line.replace('## ', '').strip(), ln=1, fill=True, align='C', border=0) # align='C'로 중앙 정렬
#             pdf.ln(3)
#             # --- 디자인 리셋 (중요!) ---
#             # 다음 텍스트에 영향이 가지 않도록 기본 색상으로 되돌린다.
#             pdf.set_fill_color(255, 255, 255) 
#             pdf.set_text_color(0, 0, 0)
#         elif line.startswith('### '):
#             pdf.set_font(font_family, 'B', size=14)
#             pdf.multi_cell(0, 10, line.replace('### ', '').strip(), ln=1)
#             pdf.ln(1)
#         elif line.startswith('* ') or line.startswith('- '):
#             pdf.set_font(font_family, '', size=11)
#             pdf.multi_cell(0, 7, f"  • {line[2:].strip()}", ln=1)
#         elif line.startswith('<정답 및 해설>'):
#              pdf.set_font(font_family, 'B', size=11)
#              pdf.multi_cell(0, 7, line, ln=1)
#         elif line == "": # 빈 줄이면 간격을 조금 띄움
#             pdf.ln(3)
#         else:
#             pdf.set_font(font_family, '', size=11)
#             pdf.multi_cell(0, 7, line, ln=1)
            

#     print("✅ [fpdf2] PDF 변환 완료.")
#     return bytes(pdf.output(dest='S'))














# --- ✨ [수정] 멀티모달 DB 생성, 로드, 업데이트 로직 완전 교체 ---

# --- ✨ [수정] 멀티모달 DB 생성, 로드, 업데이트 로직 (FAISS 통일 최종본) ---

# def is_valid_json(path):
#     try:
#         with open(path, "r", encoding="utf-8") as f:
#             json.load(f)
#         return True
#     except Exception:
#         return False


    
# def _get_metadata_path(kb_path):
#     return os.path.join(kb_path, "index_metadata.json")


# def _setup_settings(api_key: str):
#     """API 키를 받아 LlamaIndex 전역 설정을 구성하는 헬퍼 함수"""
#     Settings.llm = GoogleGenAI(model_name="models/gemini-1.5-pro-latest", api_key=api_key)
#     Settings.embed_model = GoogleGenAIEmbedding(model_name="models/embedding-001", api_key=api_key)

# def _get_document_reader():
#     """PDF와 이미지 파일 파서를 포함하는 SimpleDirectoryReader를 반환하는 헬퍼 함수"""
#     return SimpleDirectoryReader(
#         DOCS_DIR,
#         file_extractor={
#             ".pdf": PDFReader(),
#             ".png": ImageReader(parse_text=True),
#             ".jpg": ImageReader(parse_text=True),
#             ".jpeg": ImageReader(parse_text=True),
#         }
#     )

# def create_multimodal_index(kb_name: str, api_key: str):
#     """
#     [수정됨] 문서를 로드하고, FAISS 벡터 저장소와 함께 VectorStoreIndex를 생성 후 한 번에 저장합니다.
#     """
#     kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
#     persist_path = os.path.join(kb_path, "multimodal_db")
#     os.makedirs(persist_path, exist_ok=True)

#     _setup_settings(api_key)

#     documents = _get_document_reader().load_data()
#     if not documents:
#         print("⚠️ 인덱싱할 문서가 없습니다.")
#         return

#     fs = fsspec.filesystem("file")

#     # 2. FAISS 벡터 저장소 설정
#     d = 768
#     faiss_index = faiss.IndexFlatL2(d)
#     vector_store = FaissVectorStore(faiss_index=faiss_index)

#     # 3. StorageContext를 생성할 때, 위에서 만든 fs 객체를 전달
#     storage_context = StorageContext.from_defaults(
#         vector_store=vector_store, 
#         fs=fs
#     )
    
#     # 4. 변환된 노드로부터 인덱스 생성
#     index = VectorStoreIndex(storage_context=storage_context)

#     # 5. 인덱스 저장 (이제 fs에 설정된 UTF-8 인코딩을 사용하게 됨)
#     index.storage_context.persist(persist_dir=persist_path)
#     print(f"✅ '{kb_name}' 멀티모달 인덱스 생성 및 저장 완료. 경로: {persist_path}")


# def load_multimodal_query_engine(kb_name: str, api_key: str):
#     """
#     [수정됨] 지정된 경로에서 전체 인덱스를 로드하여 쿼리 엔진을 반환합니다.
#     """
#     kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
#     persist_path = os.path.join(kb_path, "multimodal_db")

#     if not os.path.exists(persist_path):
#         print(f"⚠️ '{kb_name}'에 대한 인덱스 파일이 없습니다. 새로 생성해야 합니다.")
#         # UI에서 새로 생성하도록 유도하기 위해 None 반환 또는 에러 발생
#         return None

#     _setup_settings(api_key)

#     try:
#         print(f"📂 '{kb_name}' 멀티모달 인덱스를 로드합니다...")
#         # --- ▼▼▼ 핵심 수정 부분 시작 ▼▼▼ ---
#         # 1. UTF-8 인코딩을 사용하는 파일 시스템 객체 생성
#         fs = fsspec.filesystem("file")

#         # 2. StorageContext를 불러올 때도 fs 객체를 전달
#         storage_context = StorageContext.from_defaults(persist_dir=persist_path, fs=fs)
        
#         # 3. 수정된 storage_context로 인덱스 로드
#         index = load_index_from_storage(storage_context)
#         # --- ▲▲▲ 핵심 수정 부분 끝 ▲▲▲ ---
#         print("✅ 멀티모달 인덱스 로드 성공.")
#         return index.as_query_engine(streaming=True) # 스트리밍 활성화
#     except Exception as e:
#         print(f"❌ 인덱스 로드 실패: {e}")
#         print("ℹ️ 저장된 인덱스가 손상되었거나 현재 라이브러리 버전과 호환되지 않을 수 있습니다.")
#         print("ℹ️ 기존 DB를 삭제하고 다시 생성해 보세요.")
#         return None

# # 참고: update_multimodal_index 함수는 제공된 코드의 로직이 이미 효율적이므로 그대로 사용하셔도 좋습니다.
# # 단, create와 load 함수는 위의 수정된 버전을 사용해야 합니다.
# def _get_metadata_path(kb_path):
#     return os.path.join(kb_path, "index_metadata.json")


# def _load_metadata(kb_path):
#     meta_path = _get_metadata_path(kb_path)
#     if not os.path.exists(meta_path):
#         return {}
#     try:
#         with open(meta_path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except json.JSONDecodeError:
#         return {}


# def _save_metadata(kb_path, metadata):
#     with open(_get_metadata_path(kb_path), "w", encoding="utf-8") as f:
#         json.dump(metadata, f, indent=2, ensure_ascii=False)


# def update_multimodal_index(kb_name: str, api_key: str):
#     kb_path = os.path.join(KNOWLEDGE_BASE_DIR, kb_name)
#     persist_path = os.path.join(kb_path, "multimodal_db")
#     os.makedirs(persist_path, exist_ok=True)

#     _setup_settings(api_key)

#     prev_meta = _load_metadata(kb_path)
#     all_files = [
#         os.path.join(DOCS_DIR, f)
#         for f in os.listdir(DOCS_DIR)
#         if f.lower().endswith((".pdf", ".png", ".jpg", ".jpeg"))
#     ]

#     new_or_modified = []
#     for file_path in all_files:
#         mtime = os.path.getmtime(file_path)
#         if file_path not in prev_meta or prev_meta[file_path] < mtime:
#             new_or_modified.append(file_path)

#     if not new_or_modified:
#         print("✅ 인덱스가 최신 상태입니다. 새로운 문서가 없습니다.")
#         return

#     print(f"🧩 {len(new_or_modified)}개의 새(또는 수정된) 문서를 인덱스에 추가합니다...")

#     try:
#         storage_context = StorageContext.from_defaults(persist_dir=persist_path)
#         index = load_index_from_storage(storage_context)
#         print("📂 기존 인덱스를 로드했습니다.")
#     except Exception as e:
#         print(f"⚠️ 기존 인덱스를 불러올 수 없습니다. 새로 생성합니다: {e}")
#         return create_multimodal_index(kb_name, api_key)

#     docs_to_add = SimpleDirectoryReader(
#         input_files=new_or_modified
#     ).load_data()

#     for doc in docs_to_add:
#         index.insert(doc)

#     index.storage_context.persist(persist_dir=persist_path)
#     for f in new_or_modified:
#         prev_meta[f] = os.path.getmtime(f)
#     _save_metadata(kb_path, prev_meta)
#     print(f"✅ '{kb_name}' 멀티모달 인덱스가 성공적으로 업데이트되었습니다.")


#기존 stream 반환 로직 문제를 chuck로 해결함 뭐가 뭔지.... 추가적인 이해 설명 필수
# def image_to_base64(image_file):
#     """Streamlit의 UploadedFile 객체(이미지)를 base64 문자열로 변환합니다."""
#     image_file.seek(0) # 파일 포인터를 처음으로 되돌립니다 (재사용 대비)
#     image_bytes = image_file.read()
#     image_b64 = base64.b64encode(image_bytes).decode('utf-8')
#     return image_b64


# # --- Vision 함수 1: 파일 업로드(Base64) 방식 (MIME 타입 동적 처리로 수정) ---
# def get_response_with_vision_from_file(llm: ChatGoogleGenerativeAI, image_file, question: str, system_prompt: str):
#     """RAG 검색 없이, 업로드된 이미지 파일을 직접 보고 질문에 답변하는 Gemini Vision 함수입니다."""
#     if not isinstance(llm, ChatGoogleGenerativeAI):
#         warning_text = "Warning: Image analysis is only supported by Google (Gemini) models."
#         yield AIMessageChunk(content=warning_text)
#         return
#     try:
#         # 1. 이미지를 Base64 문자열로 인코딩
#         image_b64 = image_to_base64(image_file)
#         image_b64 = image_b64.strip().replace('\n', '').replace('\r', '') #Rectification
        
#         # 2. 🟢 Streamlit의 UploadedFile 객체에서 실제 MIME 타입 가져오기
#         image_mime_type = image_file.type # 예: 'image/png' 또는 'image/jpeg'
        
#         # 3. 🟢 Data URL 형식으로 조합
#         image_data_url = f"data:{image_mime_type};base64,{image_b64}"

#         # 4. 🟢 조합된 Data URL을 메시지에 포함
#         message = HumanMessage(
#             content=[
#                 {"type": "text", "text": system_prompt},
#                 {"type": "text", "text": f"Question: {question}"},
#                 {"type": "image_url", "image_url": {"url": image_data_url}}
#             ]
#         )
#         return llm.stream([message])
#     except Exception as e:
#         error_text = f"Error processing uploaded image: {e}"
#         yield AIMessageChunk(content=error_text)
#         return


# def get_response_with_vision_from_url(llm: ChatGoogleGenerativeAI, image_url: str, question: str, system_prompt: str):
#     """
#     RAG 검색 없이, 공개 URL을 통해 이미지를 직접 분석하는 Gemini Vision 함수입니다.
#     """
#     if not isinstance(llm, ChatGoogleGenerativeAI):
#         warning_text = "Warning: Image analysis is only supported by Google (Gemini) models. Please select Google as the AI provider in the sidebar."
#         yield AIMessageChunk(content=warning_text)
#         return
    
#     if not image_url or not image_url.strip().startswith(("http://", "https://")):
#         error_text = "Error: Please provide a valid URL starting with http:// or https://."
#         yield AIMessageChunk(content=error_text)
#         return

#     message = HumanMessage(
#         content=[
#             {"type": "text", "text": system_prompt},
#             {"type": "text", "text": f"Question: {question}"},
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": image_url
#                 }
#             }
#         ]
#     )
    
#     return llm.stream([message])
# # --- 새로운 'Vision + RAG' 융합 함수 ---
# def get_fused_vision_rag_response(llm: ChatGoogleGenerativeAI, retriever, image_file, question: str, system_prompt: str):
#     """
#     1. Vision으로 이미지의 핵심 개념을 추출하고,
#     2. 추출된 개념으로 RAG 검색을 수행한 뒤,
#     3. 이미지와 검색된 문서를 종합하여 최종 답변을 생성하는 융합 파이프라인입니다.
#     """
#     # --- 1단계: 이미지에서 핵심 개념(키워드) 추출 ---
#     concept_extraction_prompt = """Analyze the provided image and identify the single most important technical concept or topic it represents. 
#     Respond with ONLY that concept phrase, in 1 to 5 words. Do not add any explanation.
#     Example responses: 'Ohm's Law', 'Kirchhoff's Current Law', 'Low Pass Filter', 'Thévenin's theorem'.
#     """
    
#     # 개념 추출 전용으로 새 LLM 인스턴스를 만듭니다 (스트림을 사용하지 않기 위함).
#     concept_extractor_llm = ChatGoogleGenerativeAI(model=llm.model, google_api_key=llm.google_api_key)
    
#     try:
#         image_b64 = image_to_base64(image_file)
#         image_mime_type = image_file.type
#         image_data_url = f"data:{image_mime_type};base64,{image_b64}"

#         concept_message = HumanMessage(
#             content=[
#                 {"type": "text", "text": concept_extraction_prompt},
#                 {"type": "image_url", "image_url": {"url": image_data_url}}
#             ]
#         )
#         # .invoke()를 사용하여 전체 응답을 한 번에 받습니다.
#         extracted_concept = concept_extractor_llm.invoke([concept_message]).content.strip()
        
#         # 터미널에 추출된 개념을 출력하여 디버깅
#         print(f"--- [Vision->RAG] 1. Extracted Concept: '{extracted_concept}' ---")

#     except Exception as e:
#         yield AIMessageChunk(content=f"Error during concept extraction from image: {e}")
#         return

#     # --- 2단계: 추출된 개념으로 RAG 문서 검색 ---
#     try:
#         retrieved_docs = retriever.invoke(extracted_concept)
#         retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
#         # 터미널에 검색된 컨텍스트 길이를 출력하여 디버깅
#         print(f"--- [Vision->RAG] 2. Retrieved Context Length: {len(retrieved_context)} characters ---")
        
#         if not retrieved_context:
#             retrieved_context = "No relevant documents found in the knowledge base."
            
#     except Exception as e:
#         yield AIMessageChunk(content=f"Error during document retrieval with RAG: {e}")
#         return

#     # --- 3단계: 이미지 + 검색된 문서를 종합하여 최종 답변 생성 ---
#     # system_prompt (예: "You are a brilliant engineering problem solver...")를 여기에 통합합니다.
#     final_generation_prompt = f"""{system_prompt}

# You MUST use the following 'Retrieved Documents' as the primary source of truth to explain the concepts related to the image in your answer. Synthesize the information from the documents and the image to provide a comprehensive and accurate response.

# [Retrieved Documents]
# {retrieved_context}
# """

#     final_message = HumanMessage(
#         content=[
#             {"type": "text", "text": final_generation_prompt},
#             {"type": "text", "text": f"User Question: {question}"},
#             {"type": "image_url", "image_url": {"url": image_data_url}}
#         ]
#     )
    
#     # 최종 답변은 사용자에게 스트리밍으로 전달
#     print("--- [Vision->RAG] 3. Generating final fused response... ---")
#     return llm.stream([final_message])
    

