# app.py (상태 초기화 버그 최종 수정본)

from PIL import Image
import nest_asyncio
nest_asyncio.apply()
from streamlit_lottie import st_lottie
import json
import streamlit as st
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



# --- 세션 상태 초기화 ---
if 'api_provider' not in st.session_state: st.session_state.api_provider = 'NVIDIA'
if 'language' not in st.session_state: st.session_state.language = 'English'
if "messages" not in st.session_state: st.session_state.messages = []
if "retriever" not in st.session_state: st.session_state.retriever = None
if "selected_kb" not in st.session_state: st.session_state.selected_kb = LANG_TEXT[st.session_state.language]['create_new_kb_option']
if "user_api_key" not in st.session_state: st.session_state.user_api_key = ""


lang = LANG_TEXT[st.session_state.language]
create_new_kb_option = lang['create_new_kb_option']
system_prompt = SYSTEM_PROMPTS[st.session_state.language]
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
def on_kb_select_change():
    st.session_state.retriever = None
    st.session_state.selected_kb = st.session_state.kb_selector
@st.cache_resource
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
    if st.session_state.retriever is None and st.session_state.selected_kb != create_new_kb_option:
        with st.spinner(f"Loading '{st.session_state.selected_kb}'..."):
            st.session_state.retriever = rag_core.load_retriever(embedder, st.session_state.selected_kb)
        if st.session_state.retriever: st.sidebar.success(f"'{st.session_state.selected_kb}' loaded.")


final_page_title = lang['page_title']
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
elif not st.session_state.retriever: st.info(lang['Knowledge_Base_Select'])
# app.py 파일 하단, 일반 RAG 채팅 로직 (else: 블록 전체)

else:
    # 현재 언어 설정에 맞는 프롬프트를 각각 가져옵니다.
    system_prompt = SYSTEM_PROMPTS[st.session_state.language]
    contextualize_q_prompt_str = CONTEXTUALIZE_Q_PROMPTS[st.session_state.language]
    # ✨ --- 올바른 디버깅 코드 시작 --- ✨
    
    # # 1. rag_core에서 했던 것과 똑같이, 질문 재구성기('history_aware_retriever')를 직접 생성합니다.
    # #    (rag_core와 langchain.chains에서 필요한 함수들을 import 해야 합니다)
    # from langchain.chains import create_history_aware_retriever
    # from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # contextualize_q_prompt = ChatPromptTemplate.from_messages([
    #     ("system", contextualize_q_prompt_str),
    #     MessagesPlaceholder("chat_history"),
    #     ("human", "{input}"),
    # ])
    # history_aware_retriever = create_history_aware_retriever(
    #     llm, st.session_state.retriever, contextualize_q_prompt
    #)
    # ✨ --- 올바른 디버깅 코드 끝 --- ✨

    # 이전 대화 기록을 LangChain이 이해하는 형태로 변환
    chat_history_for_chain = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.messages[:-1]]
    
    # 대화형 RAG 체인 생성
    conversational_rag_chain = rag_core.create_conversational_rag_chain(llm, st.session_state.retriever, SYSTEM_PROMPTS[st.session_state.language],contextualize_q_prompt_str )

    user_input = st.chat_input(lang['chat_placeholder'])
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        
        # ✨ --- 디버깅 출력 코드 시작 --- ✨
        # 1. 위에서 만든 재구성기를 직접 실행하여 결과를 확인합니다.
        #    이것이 실제로 RAG 검색에 사용될 '재구성된 질문'입니다.
        # rephrased_question_docs = history_aware_retriever.invoke({
        #     "chat_history": chat_history_for_chain,
        #     "input": user_input
        # })
        # # 2. 터미널(콘솔)에 재구성된 질문을 출력합니다.
        # #    history_aware_retriever는 문서(Document) 객체의 리스트를 반환합니다.
        # print("==============================================")
        # print(f"👤 원본 질문: {user_input}")
        # print(f"🤖 재구성 후 검색된 문서 개수: {len(rephrased_question_docs)}")
        # print("📝 검색된 문서 내용 (재구성된 질문의 결과):")
        # for i, doc in enumerate(rephrased_question_docs):
        #     print(f"--- 문서 {i+1} ---\n{doc.page_content}\n")
        # print("==============================================")
        # # ✨ --- 디버깅 출력 코드 끝 --- ✨
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            LOTTIE_FILE_PATH = "UI_Animation/Material wave loading.json"
            message_placeholder = st.empty()

            # 1. 모든 요소(Lottie, 텍스트, 최종 답변)가 그려질 단 하나의 placeholder를 만듭니다.
           
            # ✨ --- Thinking 애니메이션 로직 시작 --- ✨
            try:
                # 2. 로컬 Lottie 파일을 로드합니다. (경로 확인 필수)
                #    이 로직은 매번 실행되므로, 파일 로드 함수 위에 @st.cache_data를 붙이는 것이 성능에 좋습니다.
                lottie_thinking_json = load_lottiefile("UI_Animation/Material wave loading.json")
                
                # 3. placeholder 안에 container를 만들고, 그 안에 컬럼과 모든 요소를 배치합니다.
                with message_placeholder.container():
                    col1, col2 = st.columns([1, 6.3]) # 찾으신 최적의 비율
                    
                    with col1:
                        st_lottie(
                            lottie_thinking_json,
                            height=130,
                            width=80,
                            quality='medium',
                            key="thinking" # key는 간단하게 하나만 지정
                        )
                    

            except FileNotFoundError:
                # Lottie 파일을 찾지 못할 경우를 대비한 예외 처리
                message_placeholder.markdown("EE-Assistant is thinking... ▌")
            except Exception as e:
                # 기타 Lottie 관련 에러 발생 시
                print(f"Lottie Error: {e}")
                message_placeholder.markdown("EE-Assistant is thinking... ▌")
            # ✨ --- Thinking 애니메이션 로직 끝 --- ✨
            

            # 2. 소스(참고 자료)가 표시될 expander를 미리 만듭니다. (내용은 비어있음)
            source_expander = st.expander("참고 자료 (Source Documents)")
            source_container = source_expander.container() # expander 내부에 컨텐츠를 추가할 컨테이너
            
            full_response = ""
            
            # 3. 스피너는 이제 답변 생성 '과정 전체'가 아니라, '첫 응답이 오기 전까지'만 보여줍니다.
            #    여기서는 스피너를 제거하고, placeholder에 직접 상태를 표시하는 것이 더 좋습니다.
            # message_placeholder.markdown("EE-Assistant is thinking... :thinking:") # replaced with lottie anime 

            # 4. rag_core에서 답변과 소스를 스트리밍으로 받아옵니다.
            responses = rag_core.get_response(user_input, chat_history_for_chain, conversational_rag_chain)
            
            sources_processed = False
            for response in responses:
                # 5. 소스 처리 (단 한 번만 실행)
                if "sources" in response and not sources_processed:
                    with source_container:
                        for source in set(response["sources"]): # 중복 제거
                            st.write(f"- {source}")
                    sources_processed = True # 플래그를 설정하여 다시는 실행되지 않도록 함

                # 6. 답변 조각 처리
                if "chunk" in response:
                    full_response += response["chunk"]
                    message_placeholder.markdown(full_response + "▌")

            # 7. 스트리밍이 끝나면 커서(▌)를 제거한 최종본을 표시합니다.
            message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            # --- ✨ 개선된 로직 끝 ---
