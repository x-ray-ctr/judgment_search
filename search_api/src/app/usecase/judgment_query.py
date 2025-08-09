"""
ユースケース層 - 判例検索のアプリケーションロジック。
"""

from sentence_transformers import SentenceTransformer

from ..domain.models.judgment_dto import Judgment, JudgmentList
from ..domain.services.search_service import encode_text_to_vector
from ..infrastructure.qdrant.qdrant_gateway import query_judgments_by_vector


def handle_judgment_search(
    query: str, encoder: SentenceTransformer, limit: int = 5
) -> JudgmentList:
    """
    検索クエリに基づいて類似する判例を取得するユースケース。
    → クエリを埋め込みベクトルに変換し、Qdrantで類似チャンクを検索。
    Args:
        query: 検索したい自然言語文
        encoder: テキストをエンコードする埋め込みモデル

    Returns:
        類似判例のリスト
    """
    vector = encode_text_to_vector(query, encoder)
    results = query_judgments_by_vector(vector, limit=limit)
    return JudgmentList(items=[Judgment(**r) for r in results])
