"""
インフラ層 - Qdrant ベクトルDBへのアクセスを担うモジュール。
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List

client = QdrantClient(":memory:")


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
