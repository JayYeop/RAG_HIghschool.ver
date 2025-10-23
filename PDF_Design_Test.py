import os
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

    font_dir = "fonts"
    regular_font_path = os.path.join(font_dir, "NotoSansKR-Regular.ttf")
    bold_font_path = os.path.join(font_dir, "NotoSansKR-Bold.ttf")

    pdf = PDFWithHeaderFooter()
    font_family = "NotoSansKR"

    try:
        pdf.add_font(font_family, "", regular_font_path, uni=True)
        pdf.add_font(font_family, "B", bold_font_path, uni=True)
    except Exception as e:
        font_family = "helvetica"
    
    pdf.set_font(font_family, size=11)
    pdf.add_page()

    # [수정됨] 메인 루프를 더 단순하고 명확하게 변경
    for line in markdown_content.split('\n'):
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


# SAMPLE_MARKDOWN은 이전과 동일하게 사용하셔도 됩니다.
SAMPLE_MARKDOWN = """
# 학습 노트: 키르히호프의 전압 및 전류 법칙 (KVL/KCL)

## 📝 핵심 개념 요약
* **KCL (Kirchhoff's Current Law):** 회로의 한 노드(접합점)로 들어오는 전류의 총합은 나가는 전류의 총합과 같다. 즉, 전류의 **대수적 합은 0**이다.
* **KVL (Kirchhoff's Voltage Law):** 회로의 임의의 닫힌 루프(폐회로)를 따라 측정된 모든 전압의 대수적 합은 0이다.
* **핵심 원리:** KCL은 '**전하량 보존 법칙**'에, KVL은 '**에너지 보존 법칙**'에 근거한다.

## ✍️ 복습 퀴즈 (3문제)
**[문제 1]** KCL이 기반을 두고 있는 물리 법칙은 무엇인가요?
**[문제 2]** 아래 회로에서 저항 R2에 걸리는 전압은 얼마일까요? (단, V_source=9V, V_R1=3V)

----------------------

**[문제 1]**
<정답 및 해설>
전하량 보존 법칙입니다. 노드로 들어온 전하가 사라지거나 새로 생기지 않기 때문에, **들어온 만큼 나가야 합니다.**

**[문제 2]** 
<정답 및 해설>
KVL에 따라, 닫힌 루프의 전압 총합은 0이 되어야 합니다. 따라서 **V_source - V_R1 - V_R2 = 0** 이므로, **9V - 3V - V_R2 = 0** 입니다. V_R2는 **6V**가 됩니다.
"""

if __name__ == "__main__":
    # PDF 파일로 바로 저장하여 확인하기
    pdf_bytes = save_markdown_to_pdf(SAMPLE_MARKDOWN)
    with open("design_preview.pdf", "wb") as f:
        f.write(pdf_bytes)