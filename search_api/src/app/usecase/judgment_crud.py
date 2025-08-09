"""
ユースケース層 - 判例PDFのCRUD
"""

import uuid

from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from app.domain.services.pdf_parser import parse_pdf_into_chunks
from app.infrastructure.qdrant.qdrant_gateway import (
    delete_judgment_points,
    query_judgements_by_id,
    upsert_judgment_points,
)


def register_judgment(
    pdf_bytes: bytes, judgment_id: str, encoder: SentenceTransformer
) -> int:
    """
    Create (C in CRUD): 判例PDFをチャンクに分割してベクトル化し、Qdrantに保存する。

    Args:
        pdf_bytes (bytes): アップロードされたPDFファイルのバイナリデータ
        judgment_id (str): この判例に紐づく一意のID
        encoder (SentenceTransformer): テキストをベクトル化する埋め込みモデル

    Returns:
        int: 登録されたチャンク数（ベクトル化したテキスト断片の個数）
    """
    chunks = parse_pdf_into_chunks(pdf_bytes, max_chars_per_chunk=2000)
    if not chunks:
        return 0

    vectors = encoder.encode(chunks).tolist()
    points: list[PointStruct] = []
    for i, chunk_text in enumerate(chunks):
        doc_id = str(uuid.uuid4())
        payload = {"judgment_id": judgment_id, "chunk_index": i, "text": chunk_text}
        points.append(PointStruct(id=doc_id, vector=vectors[i], payload=payload))

    upsert_judgment_points(points)
    return len(chunks)


def read_judgment(judgment_id: str) -> list[dict]:
    """
    Read (R in CRUD): judgment_id に紐づくチャンクを全て取得する。

    Args:
        judgment_id (str): 取得対象となる判例ID

    Returns:
        List[Dict]: それぞれの要素が {"payload": dict, "score": None} のリスト
    """
    return query_judgements_by_id(judgment_id)


def update_judgment(
    pdf_bytes: bytes, judgment_id: str, encoder: SentenceTransformer
) -> int:
    """
    Update (U in CRUD): 既存の判例データを削除したうえで再登録する。

    Args:
        pdf_bytes (bytes): アップロードされたPDFのバイナリデータ
        judgment_id (str): 更新対象となる判例ID
        encoder (SentenceTransformer): テキストをベクトル化する埋め込みモデル

    Returns:
        int: 登録されたチャンク数（新たに作成されたテキスト断片の個数）
    """
    delete_judgment_points(judgment_id)
    return register_judgment(pdf_bytes, judgment_id, encoder)


def delete_judgment(judgment_id: str) -> None:
    """
    Delete (D in CRUD): 指定した判例IDに紐づくデータを削除する。

    Args:
        judgment_id (str): 削除対象となる判例ID

    Returns:
        None: 返り値は無い
    """
    delete_judgment_points(judgment_id)
