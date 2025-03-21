"""
ユースケース層 - 判例PDFの追加・更新・削除
"""
import uuid
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from typing import List
from app.domain.services.pdf_parser import parse_pdf_into_chunks
from app.infrastructure.qdrant.qdrant_gateway import (
    upsert_judgment_points,
    delete_judgment_points,
)

def register_judgment(pdf_bytes: bytes, judgment_id: str, encoder: SentenceTransformer) -> int:
    chunks = parse_pdf_into_chunks(pdf_bytes, max_chars_per_chunk=2000)
    if not chunks:
        return 0

    vectors = encoder.encode(chunks).tolist()
    points: List[PointStruct] = []
    for i, chunk_text in enumerate(chunks):
        doc_id = str(uuid.uuid4())
        payload = {
            "judgment_id": judgment_id,
            "chunk_index": i,
            "text": chunk_text
        }
        points.append(PointStruct(id=doc_id, vector=vectors[i], payload=payload))

    upsert_judgment_points(points)
    return len(chunks)

def update_judgment(pdf_bytes: bytes, judgment_id: str, encoder: SentenceTransformer) -> int:
    delete_judgment_points(judgment_id)
    return register_judgment(pdf_bytes, judgment_id, encoder)

def delete_judgment(judgment_id: str):
    delete_judgment_points(judgment_id)
