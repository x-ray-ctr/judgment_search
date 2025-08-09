"""
ドメイン層 - 判例検索結果のDTO定義。

本モジュールでは、検索APIのレスポンスとして返却される構造（1件および複数件）を定義します。
"""

from pydantic import BaseModel
from typing import List, Dict


class Judgment(BaseModel):
    """
    判例（またはその一部）を表す検索結果アイテム。

    Attributes:
        payload: 判例の内容（タイトル、本文、裁判所、年度など）
        score: クエリとの類似度スコア（例: cosine similarity）
    """
    payload: Dict
    score: float = 0.0


class JudgmentList(BaseModel):
    """
    検索結果全体を表すDTO。

    Attributes:
        items: 検索で得られた複数の判例情報
    """
    items: List[Judgment]
