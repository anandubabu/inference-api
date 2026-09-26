.PHONY: install run docker export use-model bench

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

export:
	. .venv/bin/activate && python scripts/export_model.py

# One-command bring-your-own ONNX: make use-model MODEL=./my_model.onnx
use-model:
	@test -n "$(MODEL)" || (echo 'Usage: make use-model MODEL=/path/to/model.onnx' >&2; exit 1)
	./scripts/use_model.sh "$(MODEL)"

bench:
	. .venv/bin/activate && python scripts/bench.py
