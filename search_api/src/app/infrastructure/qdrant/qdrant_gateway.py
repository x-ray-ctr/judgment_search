"""
インフラ層 - Qdrant ベクトルDBへのアクセスを担うモジュール。
"""
import os
from typing import List
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
    Distance,
)

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def create_judgement_collection():
    """ コレクション judgements を先に作成 (elasticsearchとは異なり、Qdrantでは自動作成されない) """
    if not client.collection_exists(collection_name="judgments"):
        client.create_collection(
            collection_name="judgments",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE), # 384はall-MiniLM-L6-v2のベクトルサイズ
        )
        print("コレクション judgements を作成しました。")

def query_judgments_by_vector(vector: List[float], limit: int = 5) -> List[dict]:
    """
    ベクトルに基づいてQdrantから類似判例を取得する。

    Args:
        vector: クエリベクトル
        limit: 取得する件数

    Returns:
        payload + score を含む辞書のリスト
    """
    hits = client.query_points(
        collection_name="judgments",
        query=vector,
        limit=limit
    ).points

    return [{"payload": hit.payload, "score": hit.score} for hit in hits]


def upsert_judgment_points(points: List[PointStruct]):
    """
    ポイント(ベクトル+payload)をまとめてアップサート
    - コレクション名"judgments"を固定で利用
    """
    client.upsert(collection_name="judgments", points=points)


def delete_judgment_points(judgment_id: str):
    """
    指定したjudgment_idを持つポイントを削除
    """
    client.delete(
        collection_name="judgments",
        points_selector=Filter(
            must=[FieldCondition(key="judgment_id", match=MatchValue(value=judgment_id))]
        )
    )
