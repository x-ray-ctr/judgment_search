"""
大量かつ巨大な ZIP をアップロードし、バックグラウンドでPDFをベクトル登録するルータ

- POST /judgments/upload-bulk-chunked: ZIPファイルをアップロード (multipart)
  1) ファイルを即ディスクに保存 (stream)
  2) バックグラウンドタスクで解凍・PDF抽出→チャンク化→ベクトルDB(Qdrant)登録
  3) 処理ステータスはグローバル変数upload_tasksで管理
- GET /judgments/upload-bulk-chunked/status/{task_id}: タスクの進捗を確認
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from typing import List
import os
import uuid
import shutil
import tempfile

from app.domain.services.zip_extractor import extract_pdfs_from_disk
from app.domain.services.pdf_parser import parse_pdf_into_chunks
from app.infrastructure.qdrant.qdrant_gateway import upsert_judgment_points
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

router = APIRouter()
encoder = SentenceTransformer("all-MiniLM-L6-v2")

upload_tasks = {}
MAX_ZIP_SIZE = 50 * 1024 * 1024 * 1024  # 50GB

@router.post("/judgments/upload-bulk-chunked")
async def upload_bulk_chunked(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
):
    """
    Upload a large ZIP file and process it in the background.

    1) 受け取ったZIPを一時ディレクトリへストリーミング保存
    2) バックグラウンドタスクで解凍→PDF抽出→ベクトル化→Qdrant登録
    3) 進捗はグローバル変数 `upload_tasks[task_id]` に保存

    Args:
        background_tasks (BackgroundTasks): FastAPI のバックグラウンドタスク管理
        request (Request): HTTPリクエスト、ヘッダから content-length を参照
        file (UploadFile): ZIPファイル (multipart/form-data)

    Returns:
        dict: { "task_id": str, "message": "Upload accepted. Processing in background." }

    Raises:
        HTTPException(413): ファイルサイズが 50GBを超える場合
        HTTPException(400): ファイルが空の場合
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_ZIP_SIZE:
        raise HTTPException(status_code=413, detail="File too large (>50GB)")

    task_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp(prefix=f"zip_{task_id}_")
    zip_path = os.path.join(temp_dir, "uploaded.zip")

    # ストリーミング書き込み
    with open(zip_path, "wb") as out_file:
        chunk_size = 1024 * 1024
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)

    upload_tasks[task_id] = {
        "status": "in_progress",
        "detail": "Initializing",
        "processed_pdf": 0,
        "total_pdf": 0
    }
    background_tasks.add_task(_process_zip_in_background, zip_path, task_id)

    return {"task_id": task_id, "message": "Upload accepted. Processing in background."}


@router.get("/judgments/upload-bulk-chunked/status/{task_id}")
def get_upload_status(task_id: str):
    """
    Check the progress/status of a bulk upload task by task_id.

    Args:
        task_id (str): タスクID (upload_bulk_chunked API の返却値)

    Returns:
        dict: タスクのステータス情報。例:
            {
              "status": "in_progress"|"done"|"error",
              "detail": "...",
              "processed_pdf": int,
              "total_pdf": int
            }

    Raises:
        HTTPException(404): 指定した task_id が見つからない場合
    """
    if task_id not in upload_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return upload_tasks[task_id]


def _process_zip_in_background(zip_path: str, task_id: str):
    """
    バックグラウンドでZIPを解凍→PDF抽出→チャンク化→ベクトル化→Qdrant登録。
    実行後はupload_tasks[task_id]に進捗・完了状況を格納。

    Args:
        zip_path (str): 一時保存したZIPファイルのパス
        task_id (str): バックグラウンドタスクを一意に識別するID

    Returns:
        None: (結果はグローバル変数upload_tasksで管理)

    Raises:
        Exception: 何かしらの処理中にエラーが起きた場合、status="error"をセット
    """
    try:
        upload_tasks[task_id]["status"] = "in_progress"
        upload_tasks[task_id]["detail"] = "Unpacking ZIP"

        # zipをディスクで解凍し、PDF取り出し
        pdf_files = extract_pdfs_from_disk(zip_path)

        total_pdfs = len(pdf_files)
        upload_tasks[task_id]["total_pdf"] = total_pdfs

        count_chunks = 0
        processed = 0
        for rel_path, pdf_data in pdf_files:
            processed += 1
            upload_tasks[task_id]["detail"] = f"Processing {processed}/{total_pdfs}"
            upload_tasks[task_id]["processed_pdf"] = processed

            # PDFをチャンクに分割→埋め込みベクトル化
            chunks = parse_pdf_into_chunks(pdf_data, 2000)
            if not chunks:
                continue

            vectors = encoder.encode(chunks).tolist()
            points = []
            # サブディレクトリ含むパスを judgment_id として使う
            judgment_id = rel_path.replace("/", "__").replace("\\", "__")

            for i, text_chunk in enumerate(chunks):
                pid = str(uuid.uuid4())
                points.append(PointStruct(
                    id=pid,
                    vector=vectors[i],
                    payload={
                        "judgment_id": judgment_id,
                        "chunk_index": i,
                        "text": text_chunk
                    }
                ))
            upsert_judgment_points(points)
            count_chunks += len(chunks)

        upload_tasks[task_id]["status"] = "done"
        upload_tasks[task_id]["detail"] = f"Completed. total_chunks={count_chunks}"

    except Exception as e:
        # エラーが起きたら status="error" に設定
        upload_tasks[task_id]["status"] = "error"
        upload_tasks[task_id]["detail"] = str(e)
    finally:
        # zipファイル + 解凍ディレクトリを削除
        shutil.rmtree(os.path.dirname(zip_path), ignore_errors=True)
