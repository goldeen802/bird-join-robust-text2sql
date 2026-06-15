.PHONY: install test data train eval demo
install:
	pip install -r requirements-dev.txt
test:
	pytest -q
data:
	python -m src.data.download && python -m src.data.subset && python -m src.data.build_training
eval:
	python scripts/run_eval.py --config configs/pipeline.yaml
demo:
	streamlit run app/streamlit_app.py
