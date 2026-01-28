# Project Plan - Organoid Media ML

## Project Goals
1. Predict optimal media formulations from clinical/genomic data
2. Understand gene-media relationships
3. Enable personalized organoid culture

---

## Completed Milestones

### Phase 1: Data Collection ✅
- [x] Scraped ATCC media protocols
- [x] Downloaded GDC MAF files
- [x] Built SQLite database

### Phase 2: Feature Engineering ✅
- [x] Extracted media factors from raw text
- [x] Processed VAF values for top 50 genes
- [x] Created pathway features
- [x] Built master_dataset_v2.csv (660 samples)

### Phase 3: Model Development ✅
- [x] Implemented MediaRecipeGenerator
- [x] Trained per-factor models
- [x] Cross-validation evaluation
- [x] Feature importance analysis

### Phase 4: Project Organization ✅
- [x] Reorganized from Desktop to ~/Projects/organoid-media-ml
- [x] Fixed hardcoded paths
- [x] Created centralized config
- [x] Initialized git repository

### Phase 5: Documentation ✅
- [x] Created CLAUDE.md (AI assistant guide)
- [x] Created TECHNICAL_LOG.md (system documentation)
- [x] Created PROJECT_PLAN.md (this file)
- [x] Created docs/DATA_DICTIONARY.md
- [x] Created docs/TROUBLESHOOTING.md
- [x] Created examples/predict_sample.py
- [x] Created scripts/docs/generate_data_dict.py

### Phase 6: Database Distribution ✅
- [x] Uploaded organoid_data.db (4.8GB) to Google Drive
- [x] Created scripts/download_database.py (automated download)
- [x] Added gdown dependency to requirements.txt
- [x] Updated README.md with simplified Quick Start
- [x] Verified repository completeness

### Phase 7: Database Upgrade v3.0 ✅
- [x] Created media_factors_v3 table (660 rows × 60 cols) - 13 factors with audit trail
- [x] Created top_variable_genes table (1000 genes ranked by variance)
- [x] Created gene_expression_top1000 table (490 × 1003) - wide format raw TPM
- [x] Created gene_expression_pathways table (490 × 29) - 9 pathway aggregates
- [x] Created gene_expression_markers table (490 × 26) - 25 curated markers
- [x] Created master_dataset_v3 table (660 × 206) - unified dataset
- [x] Created data_cleaning_log table (audit trail)
- [x] Created outlier_flags table (8 flagged cases)
- [x] Unit normalization applied (ng/mL, uM, mM, %)
- [x] Data cleaning: standardized missing values, cleaned categories
- [x] Created database/db_metadata.json for version tracking

### Phase 8: Beta Model Pipeline ✅
- [x] Created `beta/` module with XGBoost-based predictor
- [x] Implemented 80/20 train/test split with 5-fold cross-validation
- [x] Trained beta model on 660 samples (528 train / 132 test)
- [x] Best result: EGF R² = 0.993 on test set
- [x] Created confidence scoring system
- [x] Created input validators
- [x] Created simple API for demo predictions
- [x] Generated beta model report and data documentation

### Phase 9: MVP Planning & UI Handoff ✅
- [x] Defined MVP input/output spec (58 input fields, 24 output factors)
- [x] Chose API endpoint architecture (FastAPI backend + separate frontend)
- [x] Created `docs/MVP_SPEC.md` — full UI developer handoff document
- [x] Catalogued all 24 media factors with status (6 active, 2 coming soon, 16 in development)
- [x] Defined sample request/response JSON format
- [x] UI development handed off to collaborator

### Phase 10: FastAPI Backend ✅
- [x] Created `web/app.py` with FastAPI server wrapping `beta/api.py`
- [x] Added all 24 media factors with status (active, coming_soon, in_development)
- [x] Added support info and "early version" notice to API response
- [x] Fixed model feature ordering bug in `beta/model.py`
- [x] Added all 50 VAF gene defaults for minimal input predictions
- [x] Tested locally with curl - predictions working
- [x] Added fastapi, uvicorn, pydantic, xgboost to requirements.txt

---

## CURRENT STOP POINT

**Date:** 2026-01-28
**Status:** Phase 10 Complete: FastAPI Backend Built & Tested

**Last completed:**
- Built FastAPI server at `web/app.py`
- Fixed feature ordering bug that prevented inference
- API tested locally: `POST /predict` returns all 24 factors with grades
- Response includes: version, early notice, support email

**API Endpoints:**
- `GET /` - API info with support/feedback
- `GET /health` - Health check
- `GET /factors` - List all 24 media factors
- `POST /predict` - Predict media recipe

**Run locally:**
```bash
cd /Users/fahd838/Projects/organoid-media-ml
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```

**Next session:** Deploy to cloud (Render/Railway) and integrate with friend's UI

---

## Next Tasks

### Immediate
- [ ] Deploy API to cloud (Render/Railway/Heroku)
- [ ] Get deployment URL and share with friend for UI integration
- [ ] Integration testing: UI → API → prediction → display
- [ ] End-to-end polish

### Short-term
- [ ] Address binary classifier performance (class imbalance issue)
- [ ] Add more cancer-type-specific features (per v4_ml_training_plan.json)
- [ ] Retrain ML models using master_dataset_v3
- [ ] Add interaction features (VAF × TPM) for key genes
- [ ] Obtain ATCC formulation concentrations for 8 missing factors

### Deferred
- [ ] Clean up Desktop files (remove originals after verification)

### Future Ideas
- [ ] Deep learning on gene_expression_top1000
- [ ] Integrate new HCMI data releases
- [ ] Multi-task learning across all media factors

---

## Recurring Tasks

### Documentation Maintenance (Background)
- [ ] Regenerate DATA_DICTIONARY.md after schema changes
- [ ] Update TROUBLESHOOTING.md when new errors are discovered
- [ ] Keep PROJECT_PLAN.md current with stop points
- [ ] Review and update TECHNICAL_LOG.md quarterly

---

## Notes for Claude

When resuming work:
1. Check the "CURRENT STOP POINT" section above
2. Look at "Next Tasks" for what to work on
3. Update this file when completing tasks
4. Always verify database connection before making claims about data
