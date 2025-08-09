"""
巨大ZIPをディスク上に展開し、PDFファイルを1つずつ読み込む関数。
"""
import os
import zipfile
import shutil
from typing import List, Tuple

def extract_pdfs_from_disk(zip_path: str) -> List[Tuple[str, bytes]]:
    """
    ZIPをディスク上で展開し、各PDFをバイナリで読み込む。
    ただしファイルサイズが巨大な場合、さらに細分化やストリーミング処理を検討。
    """
    extract_dir = zip_path + "_unpacked"
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # 再帰的に .pdf ファイルを探す
    pdf_files: List[Tuple[str, bytes]] = []
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, extract_dir)
                # 読み込む(メモリに載せる): 数十GBの場合はここがネック
                # → もしさらにストリーミングしたいならpdfplumber側でファイルパス→page単位処理をする方が良い
                with open(fpath, "rb") as fp:
                    data = fp.read()
                pdf_files.append((rel_path, data))

    return pdf_files
