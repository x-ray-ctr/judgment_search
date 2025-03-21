"""
ユースケース層 - 判例検索のアプリケーションロジック。
"""

from app.domain.models.judgment_dto import JudgmentList, Judgment
from app.infrastructure.qdrant.qdrant_gateway import query_judgments_by_vector
from app.domain.services.search_service import encode_text_to_vector
from sentence_transformers import SentenceTransformer


def handle_judgment_search(query: str, encoder: SentenceTransformer) -> JudgmentList:
    """
    検索クエリに基づいて類似する判例を取得するユースケース。

    Args:
        query: 検索したい自然言語文
        encoder: テキストをエンコードする埋め込みモデル

    Returns:
        類似判例のリスト
    """
    vector = encode_text_to_vector(query, encoder)
    results = query_judgments_by_vector(vector)
    return JudgmentList(items=[Judgment(**r) for r in results])
