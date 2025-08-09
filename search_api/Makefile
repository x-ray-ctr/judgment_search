dev:
	uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

# Docker Compose コマンド
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f

# 開発環境全体を起動（Qdrant + API）
dev-full: docker-up
	sleep 5
	uv run uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

# CI/CD コマンド
lint:
	make format
	make format-check
	make type-check
	make fix


format:
	uv run black .

format-check:
	uv run black --check .

type-check:
	uv run mypy src/ --ignore-missing-imports

fix:
	uv run ruff check --fix .

test:
	uv run pytest

security-check:
	uv run bandit -r src/

ci: format-check lint type-check test security-check