.PHONY: setup download-data ingest brand-select data-pipeline taxonomy label-mode \
        eval test lint clean run

# ---- Setup: install deps, download raw data, build caches ----
setup:
	@echo "=== Installing Python dependencies ==="
	pip install -r requirements.txt
	@echo "=== Setting up config ==="
	if not exist .env copy .env.example .env
	@echo "=== Downloading raw dataset (one-time) ==="
	$(MAKE) download-data

download-data:
	@echo "Downloading Twitter Customer Support dataset from Kaggle..."
	@python scripts/download_data.py

# ---- Phase 1: Brand selection ----
brand-select:
	@echo "=== Running brand selection ==="
	@python scripts/select_brand.py

# ---- Phase 2: Data pipeline ----
data-pipeline:
	@echo "=== Running data pipeline ==="
	@python scripts/run_data_pipeline.py

# ---- Phase 3: Intent taxonomy ----
taxonomy:
	@echo "=== Generating intent taxonomy ==="
	@python scripts/build_taxonomy.py

# ---- Phase 4: Label mode (full Streamlit app, navigate to Label Mode tab) ----
label-mode:
	@echo "=== Launching Streamlit app (navigate to Label Mode tab) ==="
	@echo "Open http://localhost:8501 and click the Label Mode page."
	streamlit run src/ui/app.py

# ---- Threshold sweep on validation split ----
threshold-sweep:
	@echo "=== Running threshold sweep ==="
	@python scripts/threshold_sweep.py
# ---- Inter-annotator agreement ----
iaa:
	@echo "=== Computing inter-annotator agreement ==="
	@python scripts/compute_iaa.py
# ---- Phase 9: Evaluation harness ----
eval:
	@echo "=== Running evaluation harness ==="
	@python scripts/run_eval.py

# ---- Tests ----
test:
	@echo "=== Running tests ==="
	pytest tests/ -v

# ---- Lint ----
lint:
	@echo "=== Linting ==="
	python -m pyflakes src/ scripts/ tests/ 2>/dev/null || echo "pyflakes not installed, skipping"

# ---- Run the full Streamlit app ----
run:
	@echo "=== Launching main app ==="
	streamlit run src/ui/app.py

# ---- Clean caches and generated data ----
clean:
	@echo "=== Cleaning caches and generated data ==="
	if exist data\cache rmdir /s /q data\cache
	if exist data\gold rmdir /s /q data\gold
	if exist reports\figures rmdir /s /q reports\figures
	if exist reports\*.md del /q reports\*.md
	if exist docs\*.md del /q docs\*.md