"""
インターフェース層 - 判例CRUD APIルーター

POST /judgments      -> 新規登録 (Create)
PUT /judgments/{id}  -> 更新       (Update)
DELETE /judgments/{id} -> 削除    (Delete)
"""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sentence_transformers import SentenceTransformer

from app.infrastructure.qdrant.qdrant_gateway import query_judgements_by_id
from app.usecase.judgment_crud import (
    delete_judgment,
    register_judgment,
    update_judgment,
)
from app.usecase.judgment_query import handle_judgment_search

router = APIRouter()
encoder = SentenceTransformer("all-MiniLM-L6-v2")


@router.post("/judgments", summary="1件の判例PDFをアップロードして新規登録")
async def create_judgment(
    file: UploadFile = File(...), judgment_id: str = "default-id"
) -> dict:
    """
    Create: PDFを1件アップロードし、Qdrantの "judgments" コレクションに登録する。

    Args:
        file (UploadFile): 判例PDFファイル (multipart/form-data で送信)
        judgment_id (str): 登録したい判例の一意ID (デフォルト "default-id")

    Returns:
        dict: 例 {"message": "Created 10 chunks for judgment_id=1111"}

    Raises:
        HTTPException(400): PDFが空 or テキスト抽出できなかった場合
    """
    pdf_bytes = file.file.read()
    if not pdf_bytes:
        raise HTTPException(400, "No file data")

    num_chunks = register_judgment(pdf_bytes, judgment_id, encoder)
    if num_chunks == 0:
        raise HTTPException(400, "No text extracted from PDF")
    return {"message": f"Created {num_chunks} chunks for judgment_id={judgment_id}"}


@router.get("/judgments/search-by-vector", summary="ベクトル検索による判例取得")
def search_judgments(
    q: str = Query(..., description="検索クエリ (自然言語)"), limit: int = 5
) -> dict:
    """
    Read by vector: クエリ文字列を埋め込みベクトルに変換し、Qdrantで類似チャンクを検索。

    Args:
        q (str): 検索クエリ文字列 (自然言語)
        limit (int): 取得する件数 (デフォルト 5)

    Returns:
        dict: 例 { "items": [ { "payload": {...}, "score": 0.75 }, ... ] }

    Raises:
        HTTPException(404): 類似する結果が存在しない場合
    """
    results = handle_judgment_search(query=q, encoder=encoder, limit=limit)
    if not results.items:
        raise HTTPException(404, detail="No similar judgments found.")
    return results


@router.get("/judgments/{judgment_id}", summary="指定の判例IDのチャンクを取得")
def get_judgment_by_id(judgment_id: str) -> list[dict]:
    """
    Read by ID: 指定の判例IDに紐づくチャンクをすべて取得。

    Args:
        judgment_id (str): 取得したい判例のID

    Returns:
        List[dict]: 例 [{"payload": {...}, "score": None}, ...]

    Raises:
        HTTPException(404): 該当IDが登録されていない場合
    """
    results = query_judgements_by_id(judgment_id)
    if not results:
        raise HTTPException(404, f"No data found for judgment_id={judgment_id}")
    return results


@router.put("/judgments/{judgment_id}", summary="既存判例を更新(差し替え)")
async def modify_judgment(judgment_id: str, file: UploadFile = File(...)) -> dict:
    """
    Update: 既存の判例データを削除し、新たにPDFをアップロードして再登録する。

    Args:
        judgment_id (str): 更新対象の判例ID
        file (UploadFile): 新しいPDFファイル (multipart/form-data)

    Returns:
        dict: 例 {"message": "Updated judgment_id=xxx with 10 chunks"}

    Raises:
        HTTPException(400): PDFが空だった場合
    """
    pdf_bytes = file.file.read()
    if not pdf_bytes:
        raise HTTPException(400, "No file data")

    num_chunks = update_judgment(pdf_bytes, judgment_id, encoder)
    return {"message": f"Updated judgment_id={judgment_id} with {num_chunks} chunks"}


@router.delete("/judgments/{judgment_id}", summary="指定判例を削除")
async def remove_judgment(judgment_id: str) -> dict:
    """
    Delete: 指定の判例IDに紐づくデータをQdrantからすべて削除する。

    Args:
        judgment_id (str): 削除対象判例ID

    Returns:
        dict: 例 {"message": "Deleted judgment_id=xxx"}
    """
    delete_judgment(judgment_id)
    return {"message": f"Deleted judgment_id={judgment_id}"}
