.PHONY: install install-dev run docker train quantize bench test

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

install-dev:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -r requirements-dev.txt

run:
	. .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

train:
	. .venv/bin/activate && python scripts/train_export.py

quantize:
	. .venv/bin/activate && python scripts/quantize_onnx.py

bench:
	. .venv/bin/activate && python scripts/bench.py

test:
	. .venv/bin/activate && python -m pytest -q

# Optional: copy an external ONNX that matches the anomaly contract
use-model:
	@test -n "$(MODEL)" || (echo 'Usage: make use-model MODEL=/path/to/model.onnx' >&2; exit 1)
	./scripts/use_model.sh "$(MODEL)"
