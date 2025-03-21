"""
インターフェース層 - 判例CRUD APIルーター

POST /judgments      -> 新規登録 (Create)
PUT /judgments/{id}  -> 更新       (Update)
DELETE /judgments/{id} -> 削除    (Delete)
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from sentence_transformers import SentenceTransformer
from app.usecase.judgment_crud import register_judgment, update_judgment, delete_judgment

router = APIRouter()
encoder = SentenceTransformer("all-MiniLM-L6-v2")

@router.post("/judgments", summary="1件の判例PDFをアップロードして新規登録")
async def create_judgment(file: UploadFile = File(...), judgment_id: str = "default-id"):
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


@router.put("/judgments/{judgment_id}", summary="既存判例を更新(差し替え)")
async def modify_judgment(judgment_id: str, file: UploadFile = File(...)):
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
async def remove_judgment(judgment_id: str):
    """
    Delete: 指定の判例IDに紐づくデータをQdrantからすべて削除する。

    Args:
        judgment_id (str): 削除対象判例ID

    Returns:
        dict: 例 {"message": "Deleted judgment_id=xxx"}
    """
    delete_judgment(judgment_id)
    return {"message": f"Deleted judgment_id={judgment_id}"}
