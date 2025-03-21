"""
インターフェース層 - 判例検索APIルーター。
"""

from fastapi import APIRouter, Query
from domain.models.judgment_dto import JudgmentList
from usecase.judgment_query import handle_judgment_search
from sentence_transformers import SentenceTransformer

router = APIRouter()
encoder = SentenceTransformer("all-MiniLM-L6-v2")


@router.get("/judgments/search", response_model=JudgmentList)
def search_judgments(q: str = Query(..., description="検索クエリ")):
    """
    意味的検索を用いた類似判例の取得。

    Args:
        q: ユーザーが入力する自然言語クエリ

    Returns:
        類似する判例のリスト（類似度順）
    """
    return handle_judgment_search(q, encoder)
