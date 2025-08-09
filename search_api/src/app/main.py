"""
FastAPI アプリの起動エントリーポイント。
"""

from fastapi import FastAPI
from .interface.api.routers import judgment_router, judgment_bulk_router
from .infrastructure.qdrant.qdrant_gateway import create_judgement_collection


def create_app() -> FastAPI:
    """
    FastAPI アプリケーションを生成するファクトリ関数。

    Returns:
        アプリケーションインスタンス
    """
    app = FastAPI(title="Judgment Search API")
    # 単体PDF処理
    app.include_router(judgment_router.router, prefix="/api")
    # 大量PDF処理
    app.include_router(judgment_bulk_router.router, prefix="/api")
    # start_appイベントでコレクション作成など初期処理
    @app.on_event("startup")
    def on_startup():
        create_judgement_collection()
    return app


app = create_app()


def main():
    """
    CLI経由で起動された場合に呼ばれるメイン関数。
    開発・デバッグ用の確認メッセージを表示。
    """
    print("Hello from search-api!")


if __name__ == "__main__":
    main()

