# CareConnect - Module 5: Stakeholder Dashboard & Presentation (BAN 6800)

All numbers in the deck and dashboard come from `artifacts/*.json`, produced by the code in `analysis/`
(Module 3 pipeline -> Module 4 models -> fairness + SHAP + mitigation) run on the real Kaggle file.

## Run the dashboard locally
    pip install -r requirements.txt
    streamlit run src/app.py

## Deploy to Streamlit Cloud (required for grading)
1. Push this folder to your GitHub repo (keep `artifacts/` - the app reads it).
2. share.streamlit.io -> New app -> repo, branch, main file `src/app.py`, Python 3.11 or 3.12.
3. Copy the live URL into slides 1, 15 and A10 of `docs/CareConnect_Module5_Presentation.pptx`.

## API demo
    pip install -r requirements-api.txt
    python api_demo.py          # output saved in docs/api_demo_output.txt

## Re-run the analysis (needs the Kaggle CSV at data/raw/KaggleVeda_NO_SHOW.csv)
    python analysis/run_module3_direct.py   # Module 3 stages -> modeling parquet
    python analysis/run_real.py             # baselines + LR/RF/XGBoost
    python analysis/analyze.py              # threshold, fairness, SHAP, what-if
    python analysis/mitigate.py             # fairness mitigation options
    node analysis/build_deck.js             # rebuild the deck (pptxgenjs)

## Notes
- Tuning used a 25k-row training sample (3-fold CV) because of limited compute; final models were refit on the full training set.
- Module 3's table keeps the raw `No-show` column; it MUST be dropped before training (target leakage). `analysis/run_real.py` does this.
