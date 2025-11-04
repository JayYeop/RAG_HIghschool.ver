# app.py (상태 초기화 버그 최종 수정본)
from streamlit_extras.keyboard_url import keyboard_to_url
import pyperclip
from streamlit_extras.mention import mention
from PIL import Image
import nest_asyncio
nest_asyncio.apply()
import json
import streamlit as st
from streamlit_lottie import st_lottie
import os
import shutil
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import rag_core
from config import DOCS_DIR, KNOWLEDGE_BASE_DIR, SYSTEM_PROMPTS,LANG_TEXT,CONTEXTUALIZE_Q_PROMPTS

st.set_page_config(layout="wide", page_icon="assets/Project_logo.png")
load_dotenv()
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
# --- ▼▼▼ '질문 아이디어 보드' 기능 함수 ▼▼▼ ---

def display_pre_questions():
    """
    pre_questions.md 파일에서 추천 질문을 읽어와
    클릭 시 클립보드에 복사되는 버튼들을 생성합니다.
    """
    
    try:
        with open("pre_questions.md", "r", encoding="utf-8") as f:
            content = f.read()

        # 정규표현식을 사용하여 ### 제목과 그 아래 내용을 쌍으로 추출
        # re.DOTALL 플래그는 '.'이 줄바꿈 문자도 포함하도록 만듭니다.
        questions = re.findall(r"### (.*?)\n(.*?)(?=\n###|\Z)", content, re.DOTALL)
    
    
        with st.expander("💡 질문 아이디어"):
            st.info("아래 버튼을 클릭하면 질문이 클립보드에 복사됩니다.")

            if not questions:
                st.warning("추천 질문 파일을 찾을 수 없거나, 내용이 비어있습니다.")
                return

            cols = st.columns(2)
            for i, (label, question) in enumerate(questions):
                label = label.strip()
                question = question.strip()
                
                with cols[i % 2]:
                    # 각 질문에 대해 고유한 key를 생성해주는 것이 중요합니다.
                    if st.button(label, key=f"preq_{i}", use_container_width=True):
                        pyperclip.copy(question)
                        # 사용자에게 복사되었다는 피드백을 줍니다.
                        st.toast(f"'{label}' 질문이 복사되었습니다!", icon="📋")
                        
    except FileNotFoundError:
        # 파일이 없을 경우 경고 메시지만 표시하고 넘어갑니다.
        st.warning("'pre_questions.md' 파일을 찾을 수 없습니다.")


# --- 세션 상태 초기화 ---
if 'api_provider' not in st.session_state: st.session_state.api_provider = 'NVIDIA'
if 'language' not in st.session_state: st.session_state.language = 'English'
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None
if "selected_kb" not in st.session_state: st.session_state.selected_kb = LANG_TEXT[st.session_state.language]['create_new_kb_option']
if "user_api_key" not in st.session_state: st.session_state.user_api_key = ""
# if "multimodal_engine" not in st.session_state: st.session_state.multimodal_engine = None # ✨ 이것도 추가하면 더 좋습니다.
# if "use_multimodal" not in st.session_state: st.session_state.use_multimodal = False
# ✨ --- 수정된 부분 시작 (이 줄을 추가하세요) --- ✨

lang = LANG_TEXT[st.session_state.language]
create_new_kb_option = lang['create_new_kb_option']
system_prompt = SYSTEM_PROMPTS[st.session_state.language]
# print(system_prompt)
if "api_key_source" not in st.session_state:
    st.session_state.api_key_source = lang['api_key_source_local']
valid_api_sources = [lang['api_key_source_local'], lang['api_key_source_user']]
if st.session_state.api_key_source not in valid_api_sources:
    st.session_state.api_key_source = lang['api_key_source_local']

# --- 헬퍼 및 콜백 함수 ---
def clear_chat_and_retriever():
    """대화 기록을 모두 초기화합니다."""
    st.session_state.messages = []
    # st.session_state.retriever = None # 리트리버도 리셋하여 KB를 다시 로드하게 만듭니다.
    # st.session_state.multimodal_engine = None # 멀티모달을 썼다면 이것도 리셋 필요
    st.success("대화 기록이 초기화되었습니다, 새 대화를 시작하세요!")
def get_knowledge_bases(include_create_new=True):
    # '방(폴더)' 목록을 가져옵니다.
    db_list = [d for d in os.listdir(KNOWLEDGE_BASE_DIR) if os.path.isdir(os.path.join(KNOWLEDGE_BASE_DIR, d))]
    if include_create_new:
        # 컨시어지가 '특별 서비스'를 항상 목록 맨 앞에 추가하도록 합니다.
        return [create_new_kb_option] + db_list
    else:
        return db_list
def is_valid_kb_name(name): return re.match("^[A-Za-z0-9_-]+$", name) is not None
def on_change_reset_retriever(): st.session_state.retriever = None
def on_api_provider_change(): 
    st.session_state.retriever = None
    st.session_state.user_api_key = ""
    st.session_state.api_key_changed = True
def on_language_change(): st.session_state.messages = []
def on_kb_select_change():
    st.session_state.retriever = None
    st.session_state.selected_kb = st.session_state.kb_selector
def get_models(api_provider, user_api_key): return rag_core.load_models(api_provider, user_api_key)
def process_chat_load():
    if 'chat_file_uploader' in st.session_state and st.session_state.chat_file_uploader is not None:
        try:
            loaded_file = st.session_state.chat_file_uploader
            data = json.load(loaded_file)
            
            kb_name_from_file = data.get("knowledge_base")
            messages_from_file = data.get("messages")

            if messages_from_file is None:
                messages_from_file = data if isinstance(data, list) else []

            if not kb_name_from_file or kb_name_from_file not in get_knowledge_bases():
                st.session_state.messages = messages_from_file
                st.warning(f"Chat history loaded, but its Knowledge Base ('{kb_name_from_file}') was not found. Please select a KB manually.")
            else:
                st.session_state.selected_kb = kb_name_from_file
                st.session_state.messages = messages_from_file
                st.session_state.retriever = None
                st.success(f"Chat history and Knowledge Base '{kb_name_from_file}' are being loaded.")
                # 콜백 함수 안에서는 st.rerun()을 명시적으로 호출할 필요가 없습니다.
        except Exception as e:
            st.error(f"Failed to load or parse chat file: {e}")
@st.cache_data
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None # 파일이 없으면 None을 반환

# ================================== 1. 홀 (사이드바) ==================================

with st.sidebar:
    # (로고, 언어, AI 제공사, API 키 UI 부분은 동일)
    try:
        logo_path = "assets/Project_logo.png"
        if os.path.exists(logo_path): st.image(Image.open(logo_path),width=200)
        else: st.error("Logo file not found.")
    except Exception as e: st.error(f"Error loading logo: {e}")
    st.subheader(lang['settings_header'])
    lang_options_value = ['English', 'Korean']
    current_lang_index = lang_options_value.index(st.session_state.language)
    selected_language = st.selectbox(
        label=lang['lang_select_label'], options=lang_options_value, index=current_lang_index,
        format_func=lambda value: "English" if value == 'English' else "Korean",
        on_change=on_language_change
    )
    if selected_language != st.session_state.language:
        st.session_state.language = selected_language
        st.rerun()
    
    api_options = ['NVIDIA', 'Google']
    api_index = api_options.index(st.session_state.api_provider)
    selected_api = st.selectbox(lang['api_select_label'], api_options, index=api_index, on_change=on_api_provider_change)
    if selected_api != st.session_state.api_provider:
        st.session_state.api_provider = selected_api
        st.rerun()

    if st.session_state.api_provider == 'NVIDIA' and st.session_state.language == 'Korean':
        st.warning(lang['nvidia_korean_warning'])
    st.divider()
    st.subheader(lang['api_key_header'])
    st.radio(lang['api_key_source_label'], [lang['api_key_source_local'], lang['api_key_source_user']], key="api_key_source")
    if st.session_state.api_key_source == lang['api_key_source_user']:
        st.text_input(lang['api_key_label'].format(api_provider=st.session_state.api_provider), type="password", key="user_api_key")
    st.divider()

     # ✨ KB 선택 로직 수정 (콜백 제거, 더 직관적인 방식으로)
    kb_options = get_knowledge_bases()
    try:
        kb_index = kb_options.index(st.session_state.selected_kb)
    except ValueError:
        kb_index = 0 # st.session_state에 저장된 값이 목록에 없으면 기본값으로

    selected_kb_from_ui = st.selectbox(lang['kb_select_label'], options=kb_options, index=kb_index)

    # UI에서 선택된 값과 session_state에 저장된 값이 다를 때만 상태를 업데이트하고 rerun
    if selected_kb_from_ui != st.session_state.selected_kb:
        st.session_state.selected_kb = selected_kb_from_ui
        st.session_state.retriever = None # 리트리버 리셋
        st.rerun()
   

    # KB 관리 UI
    if st.session_state.selected_kb == create_new_kb_option:
        st.subheader(lang['new_kb_header'])
        with st.form("new_kb_form"):
            new_kb_name = st.text_input(lang['new_kb_name_label'], help=lang['new_kb_name_help'])
            uploaded_files = st.file_uploader(lang['upload_label'], accept_multiple_files=True)
            submitted = st.form_submit_button(lang['create_button'])
    elif st.session_state.selected_kb != create_new_kb_option:
        st.subheader(lang['update_kb_header'])
        with st.form("update_kb_form"):
            update_files = st.file_uploader(lang['update_upload_label'], accept_multiple_files=True)
            update_submitted = st.form_submit_button(lang['update_button'])
        st.divider()
        if st.button(lang['kb_reset_button']):
            shutil.rmtree(os.path.join(KNOWLEDGE_BASE_DIR, st.session_state.selected_kb))
            st.success(lang['kb_reset_success'].format(kb_name=st.session_state.selected_kb))
            st.session_state.selected_kb = create_new_kb_option
            st.rerun()
    st.divider()
    # # ✨ --- 수정된 부분 시작 --- ✨
    # use_multimodal = st.toggle("✨ Enable Vision DB (Multimodal RAG)", value=st.session_state.use_multimodal, help="...")
    
    # # 토글 상태가 변경되었는지 감지하는 로직 추가
    # if use_multimodal != st.session_state.use_multimodal:
    #     st.session_state.use_multimodal = use_multimodal
    #     st.session_state.retriever = None # 모든 엔진/리트리버 리셋
    #     st.session_state.multimodal_engine = None
    #     st.rerun() # 앱을 재실행하여 올바른 엔진을 로드하도록 함
    # # ✨ --- 수정된 부분 끝 --- ✨
    


    # 채팅 저장/불러오기 UI
    st.subheader(lang['chat_history_header'])

    
    
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"chat_history_{now}.json"
    
    chat_data_to_save = {
        "knowledge_base": st.session_state.selected_kb if st.session_state.selected_kb != create_new_kb_option else None,
        "messages": st.session_state.messages
    }
    chat_history_json = json.dumps(chat_data_to_save, indent=2, ensure_ascii=False)
    st.download_button(
        label=lang['chat_history_save_button'], 
        data=chat_history_json, 
        file_name=file_name, 
        mime="application/json",
        key="download_btn" # ✨ 안정성을 위해 key 추가
    )
    st.button(
    lang['chat_history_delete_button'],
    type="secondary",  # 버튼 스타일을 강조하지 않도록 설정
    on_click=clear_chat_and_retriever, # 클릭 시 정의된 함수 실행
    help=lang['chat_history_delete_button'])

    # ✨ 2. 파일 업로더에 key와 on_change 콜백을 연결합니다.
    st.file_uploader(
        label=lang['chat_history_load_label'], 
        type=['json'], 
        key='chat_file_uploader', # 위젯의 상태를 참조하기 위한 key
        on_change=process_chat_load # 파일이 업로드되면 이 함수를 실행
    )

# ================================== 2. 주방 (메인 로직) ==================================
final_api_key = None
if st.session_state.api_key_source == lang['api_key_source_local']:
    try: final_api_key = st.secrets[f"{st.session_state.api_provider.upper()}_API_KEY"]
    except: final_api_key = os.getenv(f"{st.session_state.api_provider.upper()}_API_KEY")
else: final_api_key = st.session_state.user_api_key

if "llm" not in st.session_state or "embedder" not in st.session_state or st.session_state.get("api_key_changed", False):
    with st.spinner("Loading AI models..."):
        st.session_state.llm, st.session_state.embedder = rag_core.load_models(
            st.session_state.api_provider, final_api_key
        )
    st.session_state.api_key_changed = False # 플래그 리셋

llm, embedder = st.session_state.llm, st.session_state.embedder
api_key_ok = llm is not None
if api_key_ok:
    # --- KB 생성 로직 ---
    if 'submitted' in locals() and submitted:
        if not new_kb_name or not is_valid_kb_name(new_kb_name):
            st.error(lang['invalid_kb_name_error'])
        elif not uploaded_files:
            st.warning("Please upload files.")
        else:
            if os.path.exists(DOCS_DIR): shutil.rmtree(DOCS_DIR)
            os.makedirs(DOCS_DIR)
            for file in uploaded_files:
                with open(os.path.join(DOCS_DIR, file.name), "wb") as f:
                    f.write(file.read())
            
            try:
                # if st.session_state.use_multimodal:
                #     # Vision DB 모드일 때: 멀티모달 인덱스 생성
                #     with st.spinner(f"Creating Vision DB '{new_kb_name}'..."):
                #         rag_core.create_multimodal_index(new_kb_name, final_api_key)
                #     st.success(f"Vision DB '{new_kb_name}' created.")
            
                # 텍스트 DB 모드일 때: 기존 리트리버 생성
                with st.spinner(lang['creating_db'].format(kb_name=new_kb_name)):
                    rag_core.create_and_save_retriever(llm,embedder, new_kb_name)
                st.success(lang['db_created_success'].format(kb_name=new_kb_name))
                
                st.session_state.selected_kb = new_kb_name
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create Knowledge Base: {e}")
                st.error(f"새 지식베이스를 생성하지 못하였습니다: {e}")
    # --- KB 업데이트 로직 ---
    if 'update_submitted' in locals() and update_submitted:
        if not update_files:
            st.warning("Please upload files to add.")
        else:
            # 새로 업로드된 파일을 임시 폴더에 저장
            if os.path.exists(DOCS_DIR): shutil.rmtree(DOCS_DIR)
            os.makedirs(DOCS_DIR)
            for file in update_files:
                with open(os.path.join(DOCS_DIR, file.name), "wb") as f:
                    f.write(file.read())

            try:
                # 현재 모드에 따라 올바른 업데이트 함수를 호출
                # if st.session_state.use_multimodal:
                #     # <<< 핵심 수정: 새로 만든 효율적인 업데이트 함수를 호출합니다 >>>
                #     with st.spinner(f"Updating Vision DB '{st.session_state.selected_kb}'..."):
                #         rag_core.update_multimodal_index(st.session_state.selected_kb, final_api_key)
                #     st.success(f"Vision DB '{st.session_state.selected_kb}' updated.")
                
                # 기존 텍스트 DB 업데이트 로직은 그대로 유지
                with st.spinner(lang['updating_db'].format(kb_name=st.session_state.selected_kb)):
                    st.session_state.retriever = rag_core.update_and_save_retriever(llm,embedder, st.session_state.selected_kb)
                st.success(lang['db_updated_success'].format(kb_name=st.session_state.selected_kb))

                # 작업 완료 후 임시 폴더 정리 및 앱 재실행
                if os.path.exists(DOCS_DIR): shutil.rmtree(DOCS_DIR)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update Knowledge Base: {e}")
                st.error(f"새 지식베이스를 업데이트하지 못하였습니다: {e}")
            # ✨ --- 수정된 부분 끝 --- ✨

    
        
if api_key_ok and st.session_state.selected_kb != create_new_kb_option:
    if st.session_state.api_provider == 'Google':
    #     if st.session_state.multimodal_engine is None:
    #         with st.spinner(f"Loading Vision DB '{st.session_state.selected_kb}'..."):
    #             # "로드" 전용 함수를 호출합니다.
    #             st.session_state.multimodal_engine = rag_core.load_multimodal_query_engine(
    #                 st.session_state.selected_kb, final_api_key
    #             )
    #         if st.session_state.multimodal_engine:
    #             st.sidebar.success(f"Vision DB '{st.session_state.selected_kb}' loaded.")
    #     st.session_state.retriever = None
    # # 기존 텍스트 기반 RAG 모드일 때
    # else:
        if st.session_state.retriever is None:
            with st.spinner(f"Loading Text DB '{st.session_state.selected_kb}'..."):
                st.session_state.retriever = rag_core.load_retriever(embedder, st.session_state.selected_kb)
            if st.session_state.retriever: 
                st.sidebar.success(f"Text DB '{st.session_state.selected_kb}' loaded.")
        st.session_state.multimodal_engine = None # 반대쪽 엔진은 비활성화

final_page_title = lang['page_title']


# 대화가 어느정도 진행된 후에만 버튼이 보이도록 함
# 첫 아이디어 패널
if st.session_state.language == 'Korean':

    display_pre_questions() # 메세지가 < 1 이하 일때, 패널을 보여주는 함수

# --- 학습 노트 생성 기능 ---
if len(st.session_state.messages) > 3:
    # print(len(st.session_state.messages),'######Count######') #Learning Note Count Debugging
    st.divider()
    if st.button("📋 현재까지 대화 내용으로 학습 노트 만들기"):
        
        
        st.subheader("📝 AI 생성 학습 노트 📝")
        
        with st.spinner("AI가 대화 내용을 분석하여 학습 노트를 만들고 있습니다..."):
            # 스트리밍으로 화면에 표시하고, 전체 내용은 변수에 저장
            full_markdown = st.write_stream(rag_core.stream_study_guide_optimized(llm, st.session_state.messages))

        with st.spinner("PDF 파일 변환 중..."):
            # Markdown을 PDF 바이트로 변환
            pdf_output = rag_core.save_markdown_to_pdf(full_markdown)

        # PDF 다운로드 버튼 제공
        st.download_button(
            label="📥 A4 학습 노트 다운로드 (.pdf)",
            data=pdf_output,
            file_name="ai_study_guide.pdf",
            mime="application/pdf",
        )

    st.divider()
# --- 시연용 특수 기능: 이미지 분석 (Gemini Vision Demo) 섹션 ---
if st.session_state.api_provider == 'Google':
    # lang 딕셔너리에서 직접 가져옴
    # st.write(DEMO_VISION_PROMPTS)
   with st.expander(lang['vision_expander_title']):

    # 분석 모드 UI 이름과 LANG_TEXT 내부의 프롬프트 키 매핑
    vision_mode_mapping = {
        "Smart Analysis (Vision + RAG)": "vision_prompt_smart_analysis",
        "TOEIC Grammar Expert (EE-Assistant)": "vision_prompt_toeic_expert",
        "Electrical/Electronic Engineering Problem Solver (EE-Assistant)": "vision_prompt_ee_problem_solver",
        "Image Content Describer (General Purpose)": "vision_prompt_image_describer"
    }

    # 1. 분석 모드 선택
    selected_scenario_display_name = st.selectbox(
        lang['vision_select_mode_label'],
        options=list(vision_mode_mapping.keys()),
        key="vision_scenario_selection"
    )
    selected_scenario_key = vision_mode_mapping[selected_scenario_display_name]

    # 2. 이미지 소스 선택
    vision_input_mode = st.radio(
        lang['vision_input_mode_label'],
        (lang['vision_input_mode_upload'], lang['vision_input_mode_url']),
        key="vision_input_mode",
        horizontal=True,
    )

    # 3. UI 동적 변경
    uploaded_image = None
    image_url_input = None
    if st.session_state.vision_input_mode == lang['vision_input_mode_upload'] or selected_scenario_display_name == "Smart Analysis (Vision + RAG)":
        if selected_scenario_display_name == "Smart Analysis (Vision + RAG)":
            st.info(lang['vision_smart_analysis_info'])
        
        uploaded_image = st.file_uploader(
            lang['vision_upload_image_label'],
            type=['png', 'jpg', 'jpeg'],
            key="vision_file_uploader"
        )
    else:
        image_url_input = st.text_input(
            lang['vision_url_input_label'],
            placeholder=lang['vision_url_input_placeholder']
        )

    # 4. 질문 입력
    image_question = st.text_input(
        lang['vision_question_input_label'],
        placeholder=lang['vision_question_placeholder']
    )

    # 5. 분석 시작 버튼 로직
    if st.button(lang['vision_analyze_button_label']):

    
        is_input_ready = (uploaded_image or image_url_input) and image_question and api_key_ok

        if is_input_ready:
            selected_prompt_content = lang[selected_scenario_key]
            
            image_for_session = f"data:{uploaded_image.type};base64,{rag_core.image_to_base64(uploaded_image)}"if uploaded_image else image_url_input
            
            user_request_source = f"via File: {uploaded_image.name}" if uploaded_image else "via URL"
            st.session_state.messages.append({
                "role": "user",
                "content": f"Image Analysis Request ({user_request_source}): {image_question}",
                "image": image_for_session
            })
            
            with st.chat_message("user"):
                st.markdown(f"**Image Analysis Request:**\n\n- Mode: *{selected_scenario_display_name}*\n- Question: *{image_question}*")
                st.image(image_for_session, width=300)

            with st.chat_message("assistant"):
                # --- 🚨 수정: 답변 생성 로직 단순화 ---
                full_response = ""
                sources = []
                
                with st.spinner("EE-Assistant is thinking..."): # 단순한 스피너로 변경
                    responses = None
                    
                    if selected_scenario_display_name == "Smart Analysis (Vision + RAG)":
                        if not st.session_state.retriever:
                            full_response = "Smart Analysis requires a Knowledge Base."
                        elif not uploaded_image:
                            full_response = "Smart Analysis only supports 'File Upload'."
                        else:
                            responses = rag_core.get_fused_vision_rag_response(
                                llm=llm, retriever=st.session_state.retriever,
                                image_file=uploaded_image, question=image_question,
                                system_prompt=selected_prompt_content
                            )
                    else: # 일반 Vision 모드
                        if uploaded_image:
                            responses = rag_core.get_response_with_vision_from_file(
                                llm=llm, image_file=uploaded_image,
                                question=image_question, system_prompt=selected_prompt_content
                            )
                        else:
                            responses = rag_core.get_response_with_vision_from_url(
                                llm=llm, image_url=image_url_input,
                                question=image_question, system_prompt=selected_prompt_content
                            )
                    
                    if responses:
                        for response in responses:
                            if "sources" in response:
                                sources.extend(response["sources"])
                            elif "chunk" in response:
                                full_response += response['chunk'].content
                            elif hasattr(response, 'content'): # 하위 호환
                                full_response += response.content

                # --- 루프가 끝난 후, 완성된 내용을 한 번에 표시 ---
                st.markdown(full_response)
                if sources:
                    with st.expander("참고 자료 (Source Documents)"):
                        for source in sources:
                            st.write(f"- {source}")
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        elif not api_key_ok:
            st.error(lang['vision_api_key_error_message'])
        else:
            st.warning(lang['vision_missing_input_warning'])
st.subheader(final_page_title)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message and message["image"]:
            st.image(message["image"], width=300)

if not api_key_ok: st.info(lang['api_key_missing_error'])
# <<< 핵심 수정: retriever와 multimodal_engine 둘 다 없는 경우에만 메시지 표시 >>>
elif not st.session_state.retriever: 
    if st.session_state.selected_kb != create_new_kb_option:
        # KB는 선택되었지만 로딩 중일 수 있으므로, 별도 메시지는 잠시 보류하거나 스피너와 연동
        pass
    else:
        st.info(lang['Knowledge_Base_Select'])

# --- ✨ [수정] 채팅 로직 통합 ---
if user_input := st.chat_input(lang['chat_placeholder']):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. AI 어시스턴트 답변 UI 처리 (공통)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # "Thinking" 애니메이션 (공통)
        try:
            lottie_thinking_json = load_lottiefile("UI_Animation/Material wave loading.json")
            with message_placeholder.container():
                col1, _ = st.columns([1, 6.3])
                with col1:
                    st_lottie(lottie_thinking_json, height=130, width=80, quality='medium', key="thinking_animation")
        except Exception:
            message_placeholder.markdown("EE-Assistant is thinking... ▌")

        # 참고 자료 expander (공통)
        source_expander = st.expander("참고 자료 (Source Documents)")
        source_container = source_expander.container()

        full_response = ""
        sources = []

        # 3. RAG 모드에 따라 답변 생성 로직 분기
        # Vision DB 모드
        # if st.session_state.use_multimodal and st.session_state.multimodal_engine:
        #     response_object = st.session_state.multimodal_engine.query(user_input)
        #     full_response = response_object.response
        #     source_files = [node.metadata.get('file_path', 'Unknown') for node in response_object.source_nodes]
        #     sources = [os.path.basename(source) for source in set(source_files)] # 파일 이름만 추출
        #     message_placeholder.markdown(full_response) # Vision 모드는 스트리밍이 아니므로 바로 표시

        # 텍스트 DB 모드
        if st.session_state.retriever:
            chat_history_for_chain = [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                for msg in st.session_state.messages[:-1] # 마지막 user_input은 제외
            ]
            
            conversational_rag_chain = rag_core.create_conversational_rag_chain(
                llm, st.session_state.retriever, system_prompt, CONTEXTUALIZE_Q_PROMPTS[st.session_state.language]
            )
            
            responses = rag_core.get_response(user_input, chat_history_for_chain, conversational_rag_chain)
            
            sources_processed = False
            for response in responses:
                if "sources" in response and not sources_processed:
                    sources = list(set(response["sources"]))
                    sources_processed = True
                if "chunk" in response:
                    full_response += response["chunk"]
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response) # 스트리밍 완료 후 커서 제거

        # 4. 최종 결과 및 출처 표시 (공통)
        with source_container:
            for source in sources:
                st.write(f"- {source}")
        
        # 5. 대화 기록 저장 (공통)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
# ✨ --- 수정된 부분 시작 --- ✨
# # # 기존 else 블록 전체를 이 if/elif 구조로 교체합니다.
# # # Vision DB 모드가 활성화되었고, 엔진이 준비되었을 때
# # if st.session_state.use_multimodal and st.session_state.multimodal_engine:
# #     user_input = st.chat_input("Ask about text or images in your documents...")
# #     if user_input:
# #         st.session_state.messages.append({"role": "user", "content": user_input})
# #         with st.chat_message("user"):
# #             st.markdown(user_input)
        
# #         with st.chat_message("assistant"):
# #             # [수정됨] 텍스트 RAG와 동일한 UI/UX 로직 적용
# #             message_placeholder = st.empty()

# #             # --- Thinking 애니메이션 로직 ---
# #             try:
# #                 lottie_thinking_json = load_lottiefile("UI_Animation/Material wave loading.json")
# #                 with message_placeholder.container():
# #                     col1, _ = st.columns([1, 6.3])
# #                     with col1:
# #                         st_lottie(lottie_thinking_json, height=130, width=80, quality='medium', key="thinking_vision")
# #             except Exception:
# #                 message_placeholder.markdown("EE-Assistant is thinking... ▌")

# #             # --- 소스 표시 로직 ---
# #             source_expander = st.expander("참고 자료 (Source Documents)")
# #             source_container = source_expander.container()

# #             # LlamaIndex 엔진을 호출하고 결과를 분해
# #             response_object = st.session_state.multimodal_engine.query(user_input)
            
# #             # 답변 텍스트 추출
# #             full_response = response_object.response
            
# #             # 소스 정보 추출 및 표시
# #             sources = [node.metadata.get('file_path', 'Unknown') for node in response_object.source_nodes]
# #             with source_container:
# #                 for source in set(sources): # 중복 제거
# #                     # 전체 경로 대신 파일 이름만 표시하도록 수정
# #                     st.write(f"- {os.path.basename(source)}")
            
# #             # 최종 답변 표시
# #             message_placeholder.markdown(full_response)
# #             st.session_state.messages.append({"role": "assistant", "content": full_response})

# # # 텍스트 DB 모드이고, 리트리버가 준비되었을 때
# # elif not st.session_state.use_multimodal and st.session_state.retriever:
# #     # 이 부분은 이전에 완성했던 LangChain 채팅 로직을 그대로 사용합니다.
# #     # 불필요한 디버깅 코드는 정리합니다.
# #     system_prompt = SYSTEM_PROMPTS[st.session_state.language]
# #     contextualize_q_prompt_str = CONTEXTUALIZE_Q_PROMPTS[st.session_state.language]
# #     conversational_rag_chain = rag_core.create_conversational_rag_chain(
# #         llm, st.session_state.retriever, system_prompt, contextualize_q_prompt_str
# #     )
    
# #     user_input = st.chat_input(lang['chat_placeholder'])
# #     if user_input:
# #         chat_history_for_chain = [
# #             HumanMessage(content=msg["content"]) if msg["role"] == "user" 
# #             else AIMessage(content=msg["content"]) 
# #             for msg in st.session_state.messages
# #         ]
# #         st.session_state.messages.append({"role": "user", "content": user_input})
# #         with st.chat_message("user"):
# #             st.markdown(user_input)

# #         with st.chat_message("assistant"):
# #             LOTTIE_FILE_PATH = "UI_Animation/Material wave loading.json"
# #             message_placeholder = st.empty()

# #             # 1. 모든 요소(Lottie, 텍스트, 최종 답변)가 그려질 단 하나의 placeholder를 만듭니다.
           
# #             # ✨ --- Thinking 애니메이션 로직 시작 --- ✨
# #             try:
# #                 # 2. 로컬 Lottie 파일을 로드합니다. (경로 확인 필수)
# #                 #    이 로직은 매번 실행되므로, 파일 로드 함수 위에 @st.cache_data를 붙이는 것이 성능에 좋습니다.
# #                 lottie_thinking_json = load_lottiefile("UI_Animation/Material wave loading.json")
                
# #                 # 3. placeholder 안에 container를 만들고, 그 안에 컬럼과 모든 요소를 배치합니다.
# #                 with message_placeholder.container():
# #                     col1, col2 = st.columns([1, 6.3]) # 찾으신 최적의 비율
                    
# #                     with col1:
# #                         st_lottie(
# #                             lottie_thinking_json,
# #                             height=130,
# #                             width=80,
# #                             quality='medium',
# #                             key="thinking" # key는 간단하게 하나만 지정
# #                         )
                    

# #             except FileNotFoundError:
# #                 # Lottie 파일을 찾지 못할 경우를 대비한 예외 처리
# #                 message_placeholder.markdown("EE-Assistant is thinking... ▌")
# #             except Exception as e:
# #                 # 기타 Lottie 관련 에러 발생 시
# #                 print(f"Lottie Error: {e}")
# #                 message_placeholder.markdown("EE-Assistant is thinking... ▌")
# #             # ✨ --- Thinking 애니메이션 로직 끝 --- ✨
            

# #             # 2. 소스(참고 자료)가 표시될 expander를 미리 만듭니다. (내용은 비어있음)
# #             source_expander = st.expander("참고 자료 (Source Documents)")
# #             source_container = source_expander.container() # expander 내부에 컨텐츠를 추가할 컨테이너
            
# #             full_response = ""
            
# #             # 3. 스피너는 이제 답변 생성 '과정 전체'가 아니라, '첫 응답이 오기 전까지'만 보여줍니다.
# #             #    여기서는 스피너를 제거하고, placeholder에 직접 상태를 표시하는 것이 더 좋습니다.
# #             # message_placeholder.markdown("EE-Assistant is thinking... :thinking:") # replaced with lottie anime 

# #             # 4. rag_core에서 답변과 소스를 스트리밍으로 받아옵니다.
# #             responses = rag_core.get_response(user_input, chat_history_for_chain, conversational_rag_chain)
            
# #             sources_processed = False
# #             for response in responses:
# #                 # 5. 소스 처리 (단 한 번만 실행)
# #                 if "sources" in response and not sources_processed:
# #                     with source_container:
# #                         for source in set(response["sources"]): # 중복 제거
# #                             st.write(f"- {source}")
# #                     sources_processed = True # 플래그를 설정하여 다시는 실행되지 않도록 함

# #                 # 6. 답변 조각 처리
# #                 if "chunk" in response:
# #                     full_response += response["chunk"]
# #                     message_placeholder.markdown(full_response + "▌")

# #             # 7. 스트리밍이 끝나면 커서(▌)를 제거한 최종본을 표시합니다.
# #             message_placeholder.markdown(full_response)
            
# #             st.session_state.messages.append({"role": "assistant", "content": full_response})
# #             # --- ✨ 개선된 로직 끝 ---
