# main.py (API 선택 기능이 포함된 최종 완성본)
import nest_asyncio
nest_asyncio.apply()

import streamlit as st 
import os
import pickle
from dotenv import load_dotenv

# --- AI Provider 라이브러리 임포트 ---
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- 초기 설정 및 환경 변수 로드 ---
st.set_page_config(layout="wide")
load_dotenv()

vector_store_path = "vectorstore.pkl"
DOCS_DIR = os.path.abspath("./uploaded_docs")
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# --- 세션 상태 초기화 ---
if 'api_provider' not in st.session_state:
    st.session_state['api_provider'] = 'NVIDIA'
if 'language' not in st.session_state:
    st.session_state['language'] = 'English'
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 언어별 텍스트 ---
LANG_TEXT = {
    'English': {
        'page_title': "Chat with your AI Assistant, Envie!",
        'settings_header': "Settings",
        'api_select_label': "Select AI Provider",
        'lang_select_label': "Language",
        'reset_db_button': "Reset Knowledge Base",
        'db_reset_success': "Knowledge Base has been reset.",
        'db_reset_fail': "Knowledge Base is already empty.",
        'upload_header': "Add to the Knowledge Base",
        'upload_label': "Upload a file to the Knowledge Base:",
        'upload_button': "Upload!",
        'upload_success': "File {file_name} uploaded successfully!",
        'vector_store_radio': "Use existing vector store if available",
        'vector_store_loaded': "Existing vector store loaded successfully.",
        'splitting_docs': "Splitting documents into chunks...",
        'adding_to_db': "Adding document chunks to vector database...",
        'saving_db': "Saving vector store...",
        'db_created_success': "Vector store created and saved.",
        'no_docs_warning': "No documents available to process!",
        'chat_placeholder': "Ask me anything about the documents!",
        'system_prompt': "You are a helpful AI assistant named Envie. If provided with context, use it to inform your responses. If no context is available, use your general knowledge to provide a helpful response."
    },
    'Korean': {
        'page_title': "AI 어시스턴트, Envie와 대화하기",
        'settings_header': "설정",
        'api_select_label': "AI 모델 선택",
        'lang_select_label': "언어",
        'reset_db_button': "지식 베이스 초기화",
        'db_reset_success': "지식 베이스가 초기화되었습니다.",
        'db_reset_fail': "지식 베이스가 이미 비어있습니다.",
        'upload_header': "지식 베이스에 추가하기",
        'upload_label': "지식 베이스에 파일을 업로드하세요:",
        'upload_button': "업로드!",
        'upload_success': "파일 {file_name} 업로드 성공!",
        'vector_store_radio': "기존 지식 베이스(벡터 스토어) 사용",
        'vector_store_loaded': "기존 지식 베이스를 성공적으로 불러왔습니다.",
        'splitting_docs': "문서를 청크로 분할하는 중...",
        'adding_to_db': "문서 청크를 벡터 데이터베이스에 추가하는 중...",
        'saving_db': "벡터 스토어 저장 중...",
        'db_created_success': "벡터 스토어를 생성하고 저장했습니다.",
        'no_docs_warning': "처리할 문서가 없습니다!",
        'chat_placeholder': "문서에 대해 무엇이든 물어보세요!",
        'system_prompt': "당신은 Envie라는 이름의 도움이 되는 AI 어시스턴트입니다. 컨텍스트가 제공되면 응답에 참고하세요. 컨텍스트가 없으면 일반 지식을 사용하여 유용한 답변을 제공하세요. 모든 답변은 반드시 한국어로 작성해야 합니다."
    }
}
lang = LANG_TEXT[st.session_state['language']]

# --- 모델 및 임베더 동적 로딩 함수 ---
@st.cache_resource
def load_models(api_provider):
    if api_provider == 'NVIDIA':
        if not os.getenv("NVIDIA_API_KEY"):
            st.error("NVIDIA_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            return None, None
        llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
        embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-qa-4", model_type="passage")
    elif api_provider == 'Google':
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            return None, None
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        embedder = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        return None, None
    return llm, embedder

# --- 사이드바 UI 구성 ---
with st.sidebar:
    st.subheader(lang['settings_header'])

    selected_api = st.selectbox(lang['api_select_label'], ['NVIDIA', 'Google'], index=0 if st.session_state.api_provider == 'NVIDIA' else 1, key="api_provider_selector")
    if selected_api != st.session_state.api_provider:
        st.session_state.api_provider = selected_api
        st.session_state.messages = []
        st.rerun()

    selected_language = st.selectbox(lang['lang_select_label'], ['English', 'Korean'], index=0 if st.session_state.language == 'English' else 1, key="language_selector")
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.session_state.messages = []
        st.rerun()

    if st.button(lang['reset_db_button'], key="reset_db_button"):
        if os.path.exists(vector_store_path):
            os.remove(vector_store_path)
            st.session_state.messages = []
            st.success(lang['db_reset_success'])
            st.rerun()
        else:
            st.warning(lang['db_reset_fail'])

    st.divider()

    st.subheader(lang['upload_header'])
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader(lang['upload_label'], accept_multiple_files=True, key="file_uploader")
        submitted = st.form_submit_button(lang['upload_button'])
        if uploaded_files and submitted:
            for uploaded_file in uploaded_files:
                st.success(lang['upload_success'].format(file_name=uploaded_file.name))
                with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.read())

    use_existing_vector_store = st.radio(lang['vector_store_radio'], ["Yes", "No"], horizontal=True)

# --- 모델 로드 ---
llm, document_embedder = load_models(st.session_state.api_provider)

# --- 벡터 데이터베이스 로직 ---
vectorstore = None
vector_store_exists = os.path.exists(vector_store_path)

if use_existing_vector_store == "Yes":
    if vector_store_exists:
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)
        with st.sidebar:
            st.success(lang['vector_store_loaded'])
    else:
        with st.sidebar:
            st.warning("저장된 벡터 스토어가 없습니다. 'No'를 선택하여 새로 생성해주세요.")
else:
    if document_embedder:
        with st.sidebar:
            raw_documents = DirectoryLoader(DOCS_DIR).load()
            if raw_documents:
                with st.spinner(lang['splitting_docs']):
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
                    documents = text_splitter.split_documents(raw_documents)
                with st.spinner(lang['adding_to_db']):
                    vectorstore = FAISS.from_documents(documents, document_embedder)
                with st.spinner(lang['saving_db']):
                    with open(vector_store_path, "wb") as f:
                        pickle.dump(vectorstore, f)
                st.success(lang['db_created_success'])
            else:
                st.warning(lang['no_docs_warning'], icon="⚠️")

# --- 채팅 인터페이스 ---
st.subheader(lang['page_title'])
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not llm:
    st.error(f"{st.session_state.api_provider} API 키를 확인해주세요. 채팅을 진행할 수 없습니다.")
else:
    prompt_template = ChatPromptTemplate.from_messages([("system", lang['system_prompt']),("human", "{input}")])
    chain = prompt_template | llm | StrOutputParser()
    user_input = st.chat_input(lang['chat_placeholder'])
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            if vectorstore:
                retriever = vectorstore.as_retriever(search_type="mmr")
                docs = retriever.invoke(user_input)
                context = "\n\n".join([doc.page_content for doc in docs])
                augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}\n"
            else:
                augmented_user_input = user_input
            for response in chain.stream({"input": augmented_user_input}):
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})