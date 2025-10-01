# app.py (상태 초기화 버그 최종 수정본)

from PIL import Image
import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import os
import shutil
import re
import json
from datetime import datetime
from dotenv import load_dotenv

import rag_core
from config import DOCS_DIR, KNOWLEDGE_BASE_DIR

st.set_page_config(layout="wide", page_icon="assets/Project_logo.png")
load_dotenv()
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)

CREATE_NEW_KB_OPTION = "-- 새로운 지식 베이스 만들기 --"

# --- 세션 상태 초기화 ---
if 'api_provider' not in st.session_state: st.session_state.api_provider = 'NVIDIA'
if 'language' not in st.session_state: st.session_state.language = 'English'
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None
if "selected_kb" not in st.session_state: st.session_state.selected_kb = CREATE_NEW_KB_OPTION
if "user_api_key" not in st.session_state: st.session_state.user_api_key = ""

# --- 언어별 텍스트 ---
LANG_TEXT = {
    'English': {
        'page_title': "Chat with your AI Assistant, Envie!", 'settings_header': "Settings", 'api_select_label': "Select AI Provider", 'lang_select_label': "Language", 'kb_select_label': "Select Knowledge Base", 'kb_reset_button': "Delete Selected Knowledge Base", 'kb_reset_success': "Knowledge Base '{kb_name}' has been deleted.", 'new_kb_header': "Create New Knowledge Base", 'new_kb_name_label': "Enter a name for the new Knowledge Base:", 'new_kb_name_help': "Only English letters, numbers, hyphens (-), and underscores (_) are allowed.", 'invalid_kb_name_error': "Invalid name...", 'upload_label': "Upload files...", 'create_button': "Create!", 'upload_success': "File {file_name} uploaded successfully!", 'creating_db': "Creating Knowledge Base '{kb_name}'...", 'db_created_success': "Knowledge Base '{kb_name}' created.", 'chat_placeholder': "Ask me anything...", 'system_prompt': "You are a helpful AI assistant...", 'update_kb_header': "Update Selected Knowledge Base", 'update_upload_label': "Upload additional files:", 'update_button': "Add to Knowledge Base", 'updating_db': "Adding files to '{kb_name}'...", 'db_updated_success': "Knowledge Base '{kb_name}' updated.", 'api_key_header': "Enter Your API Key", 'api_key_label': "Your {api_provider} API Key", 'api_key_help': "Your API key is not stored.", 'api_key_missing_error': "Please provide a valid API key to activate the AI.", 'chat_history_header': "Chat History", 'chat_history_save_button': "Save Chat", 'chat_history_load_label': "Load Chat", 'api_key_source_label': "API Key Source", 'api_key_source_local': "Use Local (.env/Secrets)", 'api_key_source_user': "Enter Manually",
    },
    'Korean': {
        'page_title': "AI 어시스턴트, Envie와 대화하기", 'settings_header': "설정", 'api_select_label': "AI 모델 선택", 'lang_select_label': "언어", 'kb_select_label': "지식 베이스 선택", 'kb_reset_button': "선택한 지식 베이스 삭제", 'kb_reset_success': "'{kb_name}' 지식 베이스가 삭제되었습니다.", 'new_kb_header': "새로운 지식 베이스 만들기", 'new_kb_name_label': "새 지식 베이스의 이름을 입력하세요:", 'new_kb_name_help': "이름은 영문, 숫자, 하이픈(-), 언더스코어(_)만 사용할 수 있습니다.", 'invalid_kb_name_error': "이름이 유효하지 않습니다...", 'upload_label': "새 지식 베이스에 사용할 파일을 업로드하세요:", 'create_button': "생성하기!", 'upload_success': "파일 {file_name} 업로드 성공!", 'creating_db': "'{kb_name}' 지식 베이스를 생성하는 중...", 'db_created_success': "'{kb_name}' 지식 베이스가 생성되었습니다.", 'chat_placeholder': "문서에 대해 무엇이든 물어보세요!", 'system_prompt': "당신은 Envie라는 이름의 도움이 되는 AI 어시스턴트입니다...", 'update_kb_header': "선택한 지식 베이스 업데이트", 'update_upload_label': "추가할 파일을 업로드하세요:", 'update_button': "지식 베이스에 추가", 'updating_db': "'{kb_name}'에 파일을 추가하는 중...", 'db_updated_success': "'{kb_name}' 지식 베이스가 성공적으로 업데이트되었습니다.", 'api_key_header': "API 키 입력", 'api_key_label': "{api_provider} API 키", 'api_key_help': "입력한 API 키는 저장되지 않습니다.", 'api_key_missing_error': "AI를 활성화하려면 유효한 API 키를 입력해주세요.", 'chat_history_header': "대화 기록", 'chat_history_save_button': "대화 내용 저장", 'chat_history_load_label': "대화 내용 불러오기", 'api_key_source_label': "API 키 사용 방식", 'api_key_source_local': "로컬 (.env/Secrets)", 'api_key_source_user': "직접 입력",
    }
}
lang = LANG_TEXT[st.session_state.language]
if "api_key_source" not in st.session_state:
    st.session_state.api_key_source = lang['api_key_source_local']
valid_api_sources = [lang['api_key_source_local'], lang['api_key_source_user']]
if st.session_state.api_key_source not in valid_api_sources:
    st.session_state.api_key_source = lang['api_key_source_local']

# --- 헬퍼 및 콜백 함수 ---
def get_knowledge_bases(): return [d for d in os.listdir(KNOWLEDGE_BASE_DIR) if os.path.isdir(os.path.join(KNOWLEDGE_BASE_DIR, d))]
def is_valid_kb_name(name): return re.match("^[A-Za-z0-9_-]+$", name) is not None
def on_change_reset_retriever(): st.session_state.retriever = None
def on_api_provider_change(): st.session_state.retriever = None; st.session_state.user_api_key = ""
def on_language_change(): st.session_state.messages = []

@st.cache_resource
def get_models(api_provider, user_api_key): return rag_core.load_models(api_provider, user_api_key)

# ================================== 1. 홀 (사이드바) ==================================
with st.sidebar:
    try: st.image(Image.open("assets/Project_logo.png"))
    except: pass
    st.subheader(lang['settings_header'])
    
    # [수정] selectbox에서 key를 제거하고, 위젯의 반환 값을 직접 사용하도록 변경
    # 이것이 상태 초기화 버그를 피하는 가장 확실한 방법입니다.
    
    # 1. 언어 선택 (가장 먼저 처리)
    lang_options = ['English', 'Korean']
    lang_index = lang_options.index(st.session_state.language)
    selected_language = st.selectbox(lang['lang_select_label'], lang_options, index=lang_index)
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        on_language_change()
        st.rerun() # 언어가 바뀌면 UI 텍스트가 모두 바뀌어야 하므로 rerun이 필수

    # 2. AI 제공사 선택
    api_options = ['NVIDIA', 'Google']
    api_index = api_options.index(st.session_state.api_provider)
    selected_api = st.selectbox(lang['api_select_label'], api_options, index=api_index)
    if selected_api != st.session_state.api_provider:
        st.session_state.api_provider = selected_api
        on_api_provider_change()
        # 여기서는 rerun을 하지 않음. UI가 알아서 업데이트 됨.

    st.divider()
    
    st.subheader(lang['api_key_header'])
    # radio는 key를 사용해도 비교적 안정적이므로 그대로 둡니다.
    st.radio(lang['api_key_source_label'], [lang['api_key_source_local'], lang['api_key_source_user']], key="api_key_source")
    if st.session_state.api_key_source == lang['api_key_source_user']:
        st.text_input(lang['api_key_label'].format(api_provider=st.session_state.api_provider), type="password", key="user_api_key")
    st.divider()

    kb_list = get_knowledge_bases()
    kb_options = [CREATE_NEW_KB_OPTION] + kb_list
    kb_index = kb_options.index(st.session_state.selected_kb) if st.session_state.selected_kb in kb_options else 0
    selected_kb = st.selectbox(lang['kb_select_label'], options=kb_options, index=kb_index)
    if selected_kb != st.session_state.selected_kb:
        st.session_state.selected_kb = selected_kb
        on_change_reset_retriever()

    # ... (이하 지식 베이스 UI 및 채팅 기록 UI는 이전과 동일)
    if st.session_state.selected_kb == CREATE_NEW_KB_OPTION:
        st.subheader(lang['new_kb_header'])
        with st.form("new_kb_form"):
            new_kb_name = st.text_input(lang['new_kb_name_label'], help=lang['new_kb_name_help'])
            uploaded_files = st.file_uploader(lang['upload_label'], accept_multiple_files=True)
            submitted = st.form_submit_button(lang['create_button'])
    elif st.session_state.selected_kb:
        st.subheader(lang['update_kb_header'])
        with st.form("update_kb_form"):
            update_files = st.file_uploader(lang['update_upload_label'], accept_multiple_files=True)
            update_submitted = st.form_submit_button(lang['update_button'])
        st.divider()
        if st.button(lang['kb_reset_button']):
            shutil.rmtree(os.path.join(KNOWLEDGE_BASE_DIR, st.session_state.selected_kb))
            st.success(lang['kb_reset_success'].format(kb_name=st.session_state.selected_kb))
            st.session_state.selected_kb = CREATE_NEW_KB_OPTION; st.rerun()
    st.divider()
    st.subheader(lang['chat_history_header'])
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S"); file_name = f"chat_history_{now}.json"
    chat_history_json = json.dumps(st.session_state.messages, indent=2, ensure_ascii=False)
    st.download_button(label=lang['chat_history_save_button'], data=chat_history_json, file_name=file_name, mime="application/json")
    loaded_chat_file = st.file_uploader(label=lang['chat_history_load_label'], type=['json'])

# ================================== 2. 주방 (메인 로직) ==================================
final_api_key = None
if st.session_state.api_key_source == lang['api_key_source_local']:
    try: final_api_key = st.secrets[f"{st.session_state.api_provider.upper()}_API_KEY"]
    except: final_api_key = os.getenv(f"{st.session_state.api_provider.upper()}_API_KEY")
else: final_api_key = st.session_state.user_api_key

llm, embedder = get_models(st.session_state.api_provider, final_api_key)
api_key_ok = llm is not None

if api_key_ok:
    if 'submitted' in locals() and submitted:
        if not new_kb_name or not is_valid_kb_name(new_kb_name): st.error(lang['invalid_kb_name_error'])
        elif not uploaded_files: st.warning("Please upload files.")
        else:
            if os.path.exists(DOCS_DIR): shutil.rmtree(DOCS_DIR)
            os.makedirs(DOCS_DIR)
            for file in uploaded_files:
                with open(os.path.join(DOCS_DIR, file.name), "wb") as f: f.write(file.read())
            with st.spinner(lang['creating_db'].format(kb_name=new_kb_name)):
                rag_core.create_and_save_retriever(embedder, new_kb_name)
                st.success(lang['db_created_success'].format(kb_name=new_kb_name))
                st.session_state.selected_kb = new_kb_name; st.rerun()
    if 'update_submitted' in locals() and update_submitted:
        if not update_files: st.warning("Please upload files to add.")
        else:
            if os.path.exists(DOCS_DIR): shutil.rmtree(DOCS_DIR)
            os.makedirs(DOCS_DIR)
            for file in update_files:
                with open(os.path.join(DOCS_DIR, file.name), "wb") as f: f.write(file.read())
            with st.spinner(lang['updating_db'].format(kb_name=st.session_state.selected_kb)):
                st.session_state.retriever = rag_core.update_and_save_retriever(embedder, st.session_state.selected_kb)
                st.success(lang['db_updated_success'].format(kb_name=st.session_state.selected_kb))
            if os.path.exists(DOCS_DIR): shutil.rmtree(DOCS_DIR)
    if st.session_state.retriever is None and st.session_state.selected_kb != CREATE_NEW_KB_OPTION:
        with st.spinner(f"Loading '{st.session_state.selected_kb}'..."):
            st.session_state.retriever = rag_core.load_retriever(embedder, st.session_state.selected_kb)
        if st.session_state.retriever: st.sidebar.success(f"'{st.session_state.selected_kb}' loaded.")

if loaded_chat_file:
    try: st.session_state.messages = json.load(loaded_chat_file); st.rerun()
    except Exception as e: st.error(f"Failed to load chat file: {e}")

final_page_title = lang['page_title']
if st.session_state.get('api_provider') == 'NVIDIA': final_page_title += " with NVIDIA"
elif st.session_state.get('api_provider') == 'Google': final_page_title += " with Google"
st.subheader(final_page_title)
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

if not api_key_ok: st.info(lang['api_key_missing_error'])
elif not st.session_state.retriever: st.info("Please select a Knowledge Base or create a new one.")
else:
    rag_chain = rag_core.create_rag_chain(llm, st.session_state.retriever, lang['system_prompt'])
    user_input = st.chat_input(lang['chat_placeholder'])
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            responses = rag_core.get_contextual_response(user_input, st.session_state.retriever, rag_chain)
            for response in responses:
                full_response += response; message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})