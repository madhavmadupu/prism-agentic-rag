.PHONY: install dev test lint format clean docker-up docker-down frontend

install:
	pip install -r requirements.txt
	pip install -r frontend/requirements.txt
	pre-commit install

dev:
	uvicorn src.main:app --reload --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

clean:
	rm -rf `find . -type d -name __pycache__` 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov 2>/dev/null || true

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

frontend:
	streamlit run frontend/streamlit_app.py

all: install lint typecheck test
