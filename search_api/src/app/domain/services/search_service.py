"""
ドメイン層 - 検索ロジックを提供する純粋関数群。
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


def encode_text_to_vector(text: str, model: SentenceTransformer) -> List[float]:
    """
    入力テキストをベクトルに変換する純粋関数。

    Args:
        text: クエリや文書のテキスト
        model: 使用する埋め込みモデル

    Returns:
        ベクトル表現（floatのリスト）
    """
    return model.encode(text).tolist()


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    2つのベクトル間のコサイン類似度を計算する。

    Args:
        vec_a: ベクトルA
        vec_b: ベクトルB

    Returns:
        cosine similarity（-1〜1）
    """
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
