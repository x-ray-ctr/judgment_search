"""
インフラ層 - Qdrant ベクトルDBへのアクセスを担うモジュール。

.env から環境変数 QDRANT_HOST, QDRANT_PORT を読み込み。
Qdrant ではコレクション自動作成されないため、create_judgement_collection() で初期化を行う。
"""

import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_judgement_collection() -> None:
    """
    コレクション「judgments」を事前に作成する関数。
    Qdrant は Elasticsearchのように自動作成されないため、最初に呼び出すこと。

    Args:
        なし（.envの QDRANT_HOST, QDRANT_PORT を参照）

    Returns:
        None: 特に返り値はなく、成功時はコンソールにメッセージを表示する
    """
    if not client.collection_exists(collection_name="judgments"):
        client.create_collection(
            collection_name="judgments",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print("コレクション 'judgments' を作成しました。")


def query_judgments_by_vector(vector: list[float], limit: int = 5) -> list[dict]:
    """
    ベクトルに基づいて Qdrant から類似判例を検索する。

    **ベクトル検索**のため、クエリベクトルと近いもの順に上位が返る。
    → 「検索クエリを埋め込みしたベクトル」を与えると、
       コレクション内のチャンクとの類似度が高い順に取得できる。

    Args:
        vector (List[float]): 検索クエリとして使うベクトル
        limit (int): 取得する件数 (デフォルト 5)

    Returns:
        List[Dict]:
            - 要素は {"payload": dict, "score": float} 形式
            - "score" はベクトル類似度を示す指標
    """
    hits = client.query_points(
        collection_name="judgments",
        query=vector,
        limit=limit,
    ).points

    return [{"payload": hit.payload, "score": hit.score} for hit in hits]


def query_judgements_by_id(judgement_id: str) -> list[dict]:
    """
    ID(judgment_id) に紐づくチャンクをすべて取得する。

    **主に「特定IDを持つデータをフィルタで一括取得」**するのが目的。
    → 判例ID（judgment_id）をキーとして検索し、すべてのチャンクを返す。

    Args:
        judgement_id (str): 検索対象となる判例 ID

    Returns:
        List[Dict]:
            - 要素は {"payload": dict, "score": None}
            - score は使わないため None
    """
    # 初回 scroll
    points, next_page = client.scroll(
        collection_name="judgments",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="judgment_id", match=MatchValue(value=judgement_id))
            ]
        ),
        limit=1000,
        offset=None,
    )

    all_results = []
    all_results.extend(points)

    # 次ページがあれば続ける
    while next_page is not None:
        points, next_page = client.scroll(
            collection_name="judgments",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="judgment_id", match=MatchValue(value=judgement_id)
                    )
                ]
            ),
            limit=1000,
            offset=next_page,
        )
        all_results.extend(points)

    # payload + score(=None) という形で返す
    return [{"payload": hit.payload, "score": None} for hit in all_results]


def upsert_judgment_points(points: list[PointStruct]) -> None:
    """
    ポイント(ベクトル+payload)をまとめてアップサートする。
    コレクション名「judgments」固定。

    Args:
        points (List[PointStruct]): Qdrant に登録するポイントの一覧

    Returns:
        None: 特に返り値はなく、成功時に Qdrant へデータが書き込まれる
    """
    client.upsert(collection_name="judgments", points=points)


def delete_judgment_points(judgment_id: str) -> None:
    """
    指定した judgment_id を持つポイントをすべて削除する。

    Args:
        judgment_id (str): 削除対象の判例 ID

    Returns:
        None: 返り値はなく、成功時に Qdrant からデータが削除される
    """
    client.delete(
        collection_name="judgments",
        points_selector=Filter(
            must=[
                FieldCondition(key="judgment_id", match=MatchValue(value=judgment_id))
            ]
        ),
    )
