"""
pdfplumberを利用し、PDF→テキスト抽出→(必要に応じ)チャンク分割
本番運用ではページ単位抽出も検討できるが、ここではシンプルに全文連結
"""
import pdfplumber
import io
from typing import List

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    PDFバイト列→文字列(ページ連結)
    """
    lines = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines.append(text)
    return "\n".join(lines)

def chunk_text(text: str, max_chars_per_chunk: int=2000) -> List[str]:
    """
    文字数に応じてテキストを分割する純粋関数
    """
    results = []
    start = 0
    while start < len(text):
        end = start + max_chars_per_chunk
        piece = text[start:end].strip()
        if piece:
            results.append(piece)
        start = end
    return results

def parse_pdf_into_chunks(pdf_bytes: bytes, max_chars_per_chunk: int=2000) -> List[str]:
    """
    PDFバイト列→全文抽出→チャンク分割
    """
    full_text = extract_text_from_pdf(pdf_bytes)
    if not full_text.strip():
        return []
    return chunk_text(full_text, max_chars_per_chunk)
