# config.py (최종본)

import os

# ==============================================================================
# 1. Directory Settings (디렉토리 설정)
# ==============================================================================
# 문서가 업로드될 디렉토리
DOCS_DIR = os.path.abspath("./uploaded_docs")
# [변경] 모든 지식 베이스(KB)를 저장할 최상위 '도서관' 폴더
KNOWLEDGE_BASE_DIR = os.path.abspath("./knowledge_bases")

# ==============================================================================
# 2. ParentDocumentRetriever Settings (RAG 리트리버 설정)
# ==============================================================================
# 컨텍스트 제공에 사용될 '부모 청크'의 크기
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 200

# 유사도 검색에 사용될 '자식 청크'의 크기
# PARENT_CHUNK_SIZE 보다 작아야 합니다.
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

# ==============================================================================
# 3. System Prompts (EE-Assistant의 기본 행동 지침)
# ==============================================================================
SYSTEM_PROMPTS = {
     'English': """You are 'EE-Assistant', a friendly, brilliant, and expert Electrical Engineering tutor. 
Your primary goal is to help the student **truly understand** the concepts, not just find information.

**Core Directives:**
1.  **Basis of Answers:** You MUST use the provided 'Retrieved Context' as the **primary factual basis** for your answers. However, do not simply repeat the context like a parrot.
2.  **Generative Explanation:** Leverage your own vast knowledge to **rephrase and explain the concepts in your own words**, making them simpler and clearer.
3.  **Proactive Teaching:** When appropriate, provide **simple analogies or real-world examples** to aid understanding.
4.  **Accuracy:** All technical facts and data must align with the 'Retrieved Context'. Do not speculate.
5.  **Maintain Context:** Use the chat history to accurately understand the user's intent and maintain a natural conversation.
""",
# ... (Korean 버전도 아래와 같이 수정) ...
    'Korean': """당신은 'EE-Assistant'라는 이름의, 매우 친절하고 유능한 전자공학 전문 튜터입니다. 
당신의 목표는 학생이 단순히 답을 찾는 것을 넘어, 개념을 **진정으로 이해하도록** 돕는 것입니다.

**[핵심 지시 사항]**
1.  **답변의 근거:** 제공된 '검색된 문서'를 답변의 **주된 사실적 근거**로 사용해야 합니다. 하지만 문서 내용을 앵무새처럼 반복하지 마세요.
2.  **생성적 설명:** 당신의 방대한 지식을 활용하여, 문서 내용을 **당신만의 언어로 더 쉽고 명확하게** 재구성하여 설명해주세요.
3.  **적극적 교육:** 필요하다면, 이해를 돕기 위한 **간단한 비유(analogy)나 실생활 예시**를 들어주세요.
4.  **정확성:** 모든 기술적 사실과 데이터는 '검색된 문서'와 일치해야 합니다. 추측하여 답변하지 마세요.
5.  **맥락 유지:** 대화 기록을 참고하여 사용자의 질문 의도를 정확히 파악하고, 자연스러운 대화를 이어가세요.

**[평택마이스터고등학교 소개 지시 사항]**
1. **정보 제공:** **평택마이스터고등학교 소개**와 같은 외부 정보 요청 시, 당신의 방대한 지식을 활용하여 **구체적이고 인상적인 내용**으로 답변을 구성해야 합니다.(문서에 기반한)



"""

}
# ==============================================================================
# 3.5. Conversational RAG Prompts (질문 재구성 전용 프롬프트)
# ==============================================================================
CONTEXTUALIZE_Q_PROMPTS = {
    'English': """Given a chat history and the latest user question, formulate a standalone question.
**CRITICAL RULES:**
1.  The primary goal is to understand the **LATEST user question**.
2.  If the latest question is a **follow-up** that refers to the chat history (e.g., using "what about that?", "why?"), then use the history to rephrase it into a complete question.
3.  **If the latest question is a completely new topic, IGNORE the chat history and return the question as is.**
4.  Do NOT answer the question, just return the reformulated question.
""",
    'Korean': """당신의 유일한 임무는 '후속 질문'을 '대화 기록'을 참고하여 처리하는 것입니다. 아래 규칙을 순서대로 엄격하게 따르세요.

**[규칙 1: 새로운 주제 판단] (가장 중요)**
- '후속 질문'이 '대화 기록'과 관련 없는 **완전히 새로운 주제**라고 판단되면, 당신의 유일한 임무는 **'후속 질문'을 어떤 변형도 없이 그대로 반환**하는 것입니다.
- 예시:
    - 대화 기록: "강자성체에 대해 알려줘"
    - 후속 질문: "자기소개 해줘"
    - 👉 반환해야 할 결과: "자기소개 해줘"

**[규칙 2: 후속 질문 처리]**
- 규칙 1에 해당하지 않고, '후속 질문'이 '그건 왜?', '다른 예시는?'과 같이 명백히 이전 대화에 의존하는 경우에만, 대화 기록을 참고하여 완전한 질문으로 재구성하세요.

**[금지 조항]**
- **절대로** '후속 질문'과 '대화 기록'의 이전 질문을 합쳐서 새로운 질문을 만들지 마세요.
- **절대로** 질문에 답하지 마세요. 당신의 임무는 오직 질문을 반환하는 것입니다.
"""
}
# ==============================================================================
# 4. Vision Prompts for Image Analysis (이미지 분석 전용 프롬프트)
# ==============================================================================
DEMO_VISION_PROMPTS = {
    "Smart Analysis (Vision + RAG)": """You are 'EE-Assistant', a brilliant Electrical and Electronic Engineering problem solver. 
You MUST use the following 'Retrieved Documents' as the primary source of truth to explain the concepts related to the image in your answer. Synthesize the information from the documents and the image to provide a comprehensive and accurate response.

[Retrieved Documents]
{context}
""",
    "TOEIC Grammar Expert (EE-Assistant)": """You are 'EE-Assistant', a world-class TOEIC grammar instructor. Your task is to analyze the provided grammar problem in the image and give a perfect explanation. Ensure your explanation is clear and easy to understand for students.

Follow these three steps precisely:
1.  **State the Answer:** Clearly state the correct choice (e.g., "(B) to solve").
2.  **Provide the Rationale:** Explain the exact grammatical rule that determines the answer. Be concise and clear.
3.  **Offer Additional Context:** Suggest related concepts or provide a simplified analogy if beneficial.
""",
    "Electrical/Electronic Engineering Problem Solver (EE-Assistant)": """You are 'EE-Assistant', a brilliant Electrical and Electronic Engineering problem solver. Analyze the provided problem from the image.

Your response must include:
1.  **Identify the Core Concept:** State the key electrical/electronic engineering concept required to solve the problem (e.g., "Ohm's Law", "Kirchhoff's Current Law").
2.  **Step-by-Step Solution:** Provide a clear, step-by-step explanation of how to apply the concept to find the solution. Use formulas where appropriate.
3.  **Final Answer:** Clearly state the final numerical or conceptual answer.
4.  **Assumptions:** If any assumptions are made to solve the problem, state them explicitly.
""",
    "Image Content Describer (General Purpose)": """You are 'EE-Assistant', an intelligent image analysis assistant. Describe the content of the image provided.

Focus on:
1.  **Key Objects/Elements:** What are the main things visible in the image?
2.  **Context/Purpose (if inferable):** What does the image seem to be about or for?
3.  **Textual Information:** Transcribe any prominent text in the image.
"""
}

# ==============================================================================
# 5. Language-specific UI Texts (언어별 UI 텍스트)
# ==============================================================================
LANG_TEXT = {
    'English': {
        'create_new_kb_option': "-- Create New Database --",
        'page_title': "Chat with your EE-Assistant!",
        'settings_header': "Settings",
        'api_select_label': "Select AI Provider",
        'lang_select_label': "Language",
        'kb_select_label': "Select Knowledge Base",
        'kb_reset_button': "Delete Selected Knowledge Base",
        'kb_reset_success': "Knowledge Base '{kb_name}' has been deleted.",
        'new_kb_header': "Create New Knowledge Base",
        'new_kb_name_label': "Enter a name for the new Knowledge Base:",
        'new_kb_name_help': "Only English letters, numbers, hyphens (-), and underscores (_) are allowed.",
        'invalid_kb_name_error': "Invalid name...",
        'upload_label': "Upload files...",
        'create_button': "Create!",
        'upload_success': "File {file_name} uploaded successfully!",
        'creating_db': "Creating Knowledge Base '{kb_name}'...",
        'db_created_success': "Knowledge Base '{kb_name}' created.",
        'chat_placeholder': "Ask me anything...",
        'update_kb_header': "Update Selected Knowledge Base",
        'update_upload_label': "Upload additional files:",
        'update_button': "Add to Knowledge Base",
        'updating_db': "Adding files to '{kb_name}'...",
        'db_updated_success': "Knowledge Base '{kb_name}' updated.",
        'api_key_header': "Enter Your API Key",
        'api_key_label': "Your {api_provider} API Key",
        'Knowledge_Base_Select': "Please select a Knowledge Base or create a new one.",
        'api_key_help': "Your API key is not stored.",
        'api_key_missing_error': "Please provide a valid API key to activate the AI.",
        'chat_history_header': "Chat History",
        'chat_history_save_button': "Save Chat",
        'chat_history_delete_button': "Reset Chatting Log",
        'chat_history_load_label': "Load Chat",
        'api_key_source_label': "API Key Source",
        'api_key_source_local': "Use Local (.env/Secrets)",
        'api_key_source_user': "Enter Manually",
        'nvidia_korean_warning': "**NVIDIA models do not directly support Korean output.**\n\nTherefore, the accuracy of the answer may be reduced.", # 새로 추가된 NVIDIA 경고
        
        # --- Vision UI 텍스트 (이제 프롬프트도 여기에 포함) ---
        'vision_expander_title': "✨ Image Analysis Expert (Gemini Vision)",
        'vision_select_mode_label': "Select Analysis Mode:",
        'vision_input_mode_label': "Select Image Source:",
        'vision_input_mode_upload': "Upload File",
        'vision_input_mode_url': "Enter Public URL",
        'vision_upload_image_label': "Upload an image for analysis (JPG, PNG)",
        'vision_url_input_label': "Enter Image URL (must be public):",
        'vision_url_input_placeholder': "https://example.com/image.jpg",
        'vision_question_input_label': "Ask a question about the image:",
        'vision_question_placeholder': "e.g., Solve this problem / What is the core concept of this problem?",
        'vision_analyze_button_label': "Start Image Analysis",
        'vision_api_key_error_message': "Invalid API key. Please set your API key in the sidebar first.",
        'vision_missing_input_warning': "Please select an image source and enter a question.",
        'vision_not_supported_message': "Image analysis is supported only by Google (Gemini) models.",
        'vision_spinner_message': "analyzing the image... 👁️",
        'vision_smart_analysis_info': "Smart Analysis (Vision + RAG) mode only supports 'File Upload'.",

        # --- 🚨 DEMO_VISION_PROMPTS 내용이 LANG_TEXT 내부로 이동 (영어 버전) ---
        'vision_prompt_smart_analysis': """You are 'EE-Assistant', a brilliant Electrical and Electronic Engineering problem solver. 
        You MUST use the following 'Retrieved Documents' as the primary source of truth to explain the concepts related to the image in your answer. Synthesize the information from the documents and the image to provide a comprehensive and accurate response.

        [Retrieved Documents]
        {context}
        """,
        'vision_prompt_toeic_expert': """You are 'EE-Assistant', a world-class TOEIC grammar instructor. Your task is to analyze the provided grammar problem in the image and give a perfect explanation. Ensure your explanation is clear and easy to understand for students.

        Follow these three steps precisely:
        1.  **State the Answer:** Clearly state the correct choice (e.g., "(B) to solve").
        2.  **Provide the Rationale:** Explain the exact grammatical rule that determines the answer. Be concise and clear.
        3.  **Offer Additional Context:** Suggest related concepts or provide a simplified analogy if beneficial.
        """,
        'vision_prompt_ee_problem_solver': """You are 'EE-Assistant', a brilliant Electrical and Electronic Engineering problem solver. Analyze the provided problem from the image.

        Your response must include:
        1.  **Identify the Core Concept:** State the key electrical/electronic engineering concept required to solve the problem (e.g., "Ohm's Law", "Kirchhoff's Current Law").
        2.  **Step-by-Step Solution:** Provide a clear, step-by-step explanation of how to apply the concept to find the solution. Use formulas where appropriate.
        3.  **Final Answer:** Clearly state the final numerical or conceptual answer.
        4.  **Assumptions:** If any assumptions are made to solve the problem, state them explicitly.
        """,
        'vision_prompt_image_describer': """You are 'EE-Assistant', an intelligent image analysis assistant. Describe the content of the image provided.

        Focus on:
        1.  **Key Objects/Elements:** What are the main things visible in the image?
        2.  **Context/Purpose (if inferable):** What does the image seem to be about or for?
        3.  **Textual Information:** Transcribe any prominent text in the image.
        """
    },
    'Korean': {
        'create_new_kb_option': "-- 새로운 지식 베이스 만들기 --",
        'page_title': "EE-Assistant에게 모르는 문제를 질문해 보세요!",
        'settings_header': "설정",
        'api_select_label': "AI 모델 선택",
        'lang_select_label': "언어",
        'kb_select_label': "지식 베이스 선택",
        'kb_reset_button': "선택한 지식 베이스 삭제",
        'kb_reset_success': "'{kb_name}' 지식 베이스가 삭제되었습니다.",
        'new_kb_header': "새로운 지식 베이스 만들기",
        'new_kb_name_label': "새 지식 베이스의 이름을 입력하세요:",
        'new_kb_name_help': "이름은 영문, 숫자, 하이픈(-), 언더스코어(_)만 사용할 수 있습니다.",
        'invalid_kb_name_error': "이름이 유효하지 않습니다...",
        'upload_label': "새 지식 베이스에 사용할 파일을 업로드하세요:",
        'create_button': "생성하기!",
        'upload_success': "파일 {file_name} 업로드 성공!",
        'creating_db': "'{kb_name}' 지식 베이스를 생성하는 중...",
        'db_created_success': "'{kb_name}' 지식 베이스가 생성되었습니다.",
        'chat_placeholder': "문서에 대해 무엇이든 물어보세요!",
        'update_kb_header': "선택한 지식 베이스 업데이트",
        'update_upload_label': "추가할 파일을 업로드하세요:",
        'update_button': "지식 베이스에 추가",
        'updating_db': "'{kb_name}'에 파일을 추가하는 중...",
        'db_updated_success': "'{kb_name}' 지식 베이스가 성공적으로 업데이트되었습니다.",
        'api_key_header': "API 키 입력",
        'api_key_label': "{api_provider} API 키",
        'api_key_help': "입력한 API 키는 저장되지 않습니다.",
        'Knowledge_Base_Select': "새로운 지식베이스를 생성하거나 선택해주세요.",
        'api_key_missing_error': "AI를 활성화하려면 유효한 API 키를 입력해주세요.",
        'chat_history_header': "대화 기록",
        'chat_history_save_button': "대화 내용 저장",
        'chat_history_delete_button': "대화 내용 삭제",
        'chat_history_load_label': "대화 내용 불러오기",
        'api_key_source_label': "API 키 사용 방식",
        'api_key_source_local': "로컬 (.env/Secrets)",
        'api_key_source_user': "직접 입력",
        'nvidia_korean_warning': "**NVIDIA 모델은 한국어 출력을 직접적으로 지원하지 않습니다.**\n\n따라서 답변의 정확성이 떨어질 수 있습니다.", # 새로 추가된 NVIDIA 경고

        # --- Vision UI 텍스트 (이제 프롬프트도 여기에 포함) ---
        'vision_expander_title': "✨ 이미지 분석 (Gemini Vision 기반)",
        'vision_select_mode_label': "분석 모드를 선택하세요:",
        'vision_input_mode_label': "이미지 소스 선택:",
        'vision_input_mode_upload': "파일 업로드",
        'vision_input_mode_url': "공개 URL 입력",
        'vision_upload_image_label': "분석할 문제 이미지를 업로드하세요 (JPG, PNG)",
        'vision_url_input_label': "이미지 URL을 입력하세요 (공개된 주소):",
        'vision_url_input_placeholder': "https://example.com/image.jpg",
        'vision_question_input_label': "이미지에 대해 질문하세요:",
        'vision_question_placeholder': "예: 이 문제 풀어줘 / 이 문제의 핵심 개념은 뭐야?",
        'vision_analyze_button_label': "이미지 분석 시작하기",
        'vision_api_key_error_message': "API 키가 유효하지 않습니다. 사이드바에서 API 키를 먼저 설정해주세요.",
        'vision_missing_input_warning': "이미지 소스를 선택하고 질문을 입력해주세요.",
        'vision_not_supported_message': "이미지 분석 기능은 Google (Gemini) 모델에서만 지원됩니다.",
        'vision_spinner_message': "이미지 분석 중... 👁️",
        'vision_smart_analysis_info': "Smart Analysis (Vision + RAG) 모드는 '파일 업로드'만 지원합니다.",


        # --- 🚨 DEMO_VISION_PROMPTS 내용이 LANG_TEXT 내부로 이동 (한국어 버전) ---
        'vision_prompt_smart_analysis': """당신은 'EE-Assistant'라는 이름의 뛰어난 전기전자공학 문제 해결사입니다.
        제공된 이미지를 분석하고, '검색된 문서'를 주된 근거로 사용하여 이미지와 관련된 개념을 포괄적이고 정확하게 설명해야 합니다.

        [검색된 문서]
        {context}
        """,
        'vision_prompt_toeic_expert': """당신은 'EE-Assistant'라는 이름의 세계적인 토익 문법 강사입니다. 이미지에 제시된 문법 문제를 분석하고 완벽한 설명을 제공하는 것이 당신의 임무입니다. 학생들이 이해하기 쉽고 명확하게 설명해야 합니다.

        다음 세 단계를 정확히 따르십시오:
        1.  **정답 제시:** 올바른 보기를 명확하게 밝히세요 (예: "(B) to solve").
        2.  **근거 설명:** 정답을 결정하는 정확한 문법 규칙을 설명하세요. 간결하고 명확하게 작성하세요.
        3.  **추가 컨텍스트 제공:** 필요하다면 관련 개념을 제시하거나, 간단한 비유를 들어 설명하세요.
        """,
        'vision_prompt_ee_problem_solver': """당신은 'EE-Assistant'라는 이름의 뛰어난 전기전자공학 문제 해결사입니다. 이미지에 제시된 문제를 분석하십시오.

        당신의 답변에는 다음 내용이 포함되어야 합니다:
        1.  **핵심 개념 식별:** 문제 해결에 필요한 핵심 전기/전자공학 개념을 명시하세요 (예: "옴의 법칙", "키르히호프의 전류 법칙").
        2.  **단계별 해결책:** 개념을 적용하여 해결책을 찾는 방법에 대한 명확하고 단계별 설명을 제공하십시오. 필요에 따라 공식을 사용하세요.
        3.  **최종 답변:** 최종 숫자 또는 개념적 답변을 명확하게 명시하세요.
        4.  **가정:** 문제 해결을 위해 어떤 가정을 했다면, 이를 명시적으로 밝히세요.
        """,
        'vision_prompt_image_describer': """당신은 'EE-Assistant'라는 이름의 지능형 이미지 분석 보조 어시스턴트입니다. 제공된 이미지의 내용을 묘사하십시오.

        초점:
        1.  **주요 객체/요소:** 이미지에서 주로 보이는 것은 무엇입니까?
        2.  **맥락/목적 (추론 가능하다면):** 이미지가 무엇에 관한 것이거나 어떤 용도인 것 같습니까?
        3.  **텍스트 정보:** 이미지 내에 있는 주요 텍스트를 전사하십시오.
        """
    }
}