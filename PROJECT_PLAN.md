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

---

## CURRENT STOP POINT

**Date:** 2026-01-21
**Status:** Phase 6 Complete: Database distribution system established

**Last completed:**
- Reviewed repository to confirm organoid_data.db is the only missing file
- Uploaded database to Google Drive (File ID: 1B-E9pScJRukGVa9Tckc-JxMWDCw_AhhB)
- Created automated download script (scripts/download_database.py)
- Updated requirements.txt with gdown dependency
- Simplified README.md Quick Start section
- Pushed changes to GitHub (commit 6c7ceb8)

**Repository now fully self-contained:** Users can clone and run `python scripts/download_database.py` to get started.

**Next session:** Data Management

---

## Next Tasks

### Immediate (Next Session)
- [x] Review system integrity and verify all components
- [ ] Clean up Desktop files (remove originals after verification)

### Short-term
- [ ] Add unit tests for preprocessing
- [ ] Create inference API for predictions
- [ ] Document model interpretation findings

### Future Ideas
- [ ] Web interface for predictions
- [ ] Add more media factors
- [ ] Integrate new HCMI data releases
- [ ] Experiment with deep learning approaches

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
