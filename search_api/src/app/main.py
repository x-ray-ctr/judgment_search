"""
FastAPI アプリの起動エントリーポイント。
"""

from fastapi import FastAPI
from interface.api.routers import judgment_router


def create_app() -> FastAPI:
    """
    FastAPI アプリケーションを生成するファクトリ関数。

    Returns:
        アプリケーションインスタンス
    """
    app = FastAPI(title="Judgment Search API")
    app.include_router(judgment_router.router, prefix="/api")
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

