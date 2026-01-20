# CLAUDE.md - AI Assistant Guide

> **IMPORTANT:** This file helps Claude Code understand this project.
> Read this file at the start of each session.

---

## GOLDEN RULES

1. **NO FABRICATION** - Never invent data, statistics, or results. If data doesn't exist, say so.
2. **NO FAKE DATA** - All numbers must come from actual database queries or code outputs.
3. **VERIFY BEFORE CLAIM** - Run queries/code to confirm facts before stating them.

---

## First Steps for Every Session

1. **Read `TECHNICAL_LOG.md`** - Understand how the system works
2. **Read `PROJECT_PLAN.md`** - Find the current stop point and next tasks
3. Ask the user what they want to work on today

---

## Workflow Protocol (Mandatory for All Tasks)

### Phase A: Planning
Before writing any code, the Head Agent must:
1. Present a **Flow Diagram** of the proposed solution
2. Identify which specialized agents are needed
3. Get user approval before proceeding

```
Example Flow Diagram:

[User Request]
     |
[Head Agent: Analyze & Plan]
     |
[DataAgent: Query DB] -> [BioAgent: Validate Biology] -> [ModelAgent: Run Predictions]
     |
[Head Agent: QC Review]
     |
[Deliver to User]
```

### Phase B: Distribution
Define and deploy specialized agents for the task:

| Agent | Role | Tools |
|-------|------|-------|
| **Head Agent** | Coordination, QC, user communication | All |
| **DataAgent** | Database queries, data validation | SQL, Pandas |
| **BioAgent** | Biological validation, domain knowledge | Literature, ontologies |
| **ModelAgent** | ML predictions, model evaluation | sklearn, joblib |
| **DocsAgent** | Documentation updates | Markdown |

### Phase C: Review
Before final delivery, Head Agent must:
1. **QC Check** - Verify outputs from all agents
2. **Cross-validate** - Ensure data consistency across agents
3. **Audit trail** - Document what each agent did
4. **User sign-off** - Present summary for approval

---

## Standard Operating Procedure (SOP)

This project follows a strict Agent Execution Framework:

```
+-------------------------------------------------------------+
|                    HEAD AGENT (Claude)                       |
|  - Receives user request                                     |
|  - Creates flow diagram (Phase A)                           |
|  - Distributes to specialized agents (Phase B)              |
|  - Performs QC review (Phase C)                             |
|  - Delivers verified results                                |
+-------------------------------------------------------------+
                           |
     +-------------+-------------+-------------+
     |             |             |             |
+---------+  +---------+  +---------+  +---------+
|DataAgent|  |BioAgent |  |ModelAgent| |DocsAgent|
+---------+  +---------+  +---------+  +---------+
```

**Enforcement:** All significant tasks must follow this protocol.
**Exceptions:** Simple queries or single-file edits may skip Phase B.

---

## Project Overview

**Organoid Media Recipe Generator** - ML system that predicts optimal culture media
formulations for organoid models based on clinical and genomic data.

## Key Locations

| What | Where |
|------|-------|
| Database (4.5GB) | `database/organoid_data.db` |
| Training data | `data/processed/master_dataset_v2.csv` |
| Trained model | `model_artifacts/media_recipe_generator.joblib` |
| Main training script | `train.py` |
| Path configuration | `config/paths.py` |

## Quick Commands

```bash
# Verify database connection
python scripts/verification/verify_db_link.py

# Run training
python train.py

# Test imports
python -c "from config.paths import DB_PATH; print(DB_PATH)"

# Regenerate data dictionary
python scripts/docs/generate_data_dict.py
```

## Architecture Summary

```
Clinical Data + Genomic VAF -> Preprocessing -> Per-Factor Models -> Media Recipe
(660 samples)      (Top 50 genes)     (8 factors)
```

## Key Technical Details

- **Masked Training:** Each factor model trains only on non-NULL samples
- **8 Media Factors:** egf, y27632, n_acetyl_cysteine, a83_01, sb202190, fgf2, cholera_toxin, insulin
- **Model Types:** Regression (egf, cholera_toxin) and Binary Classification (others)

---

## Documentation Maintenance

The docs are kept in sync automatically. When schema changes:

```bash
# Regenerate data dictionary
python scripts/docs/generate_data_dict.py
```

**Files that auto-update:**
- `docs/DATA_DICTIONARY.md` - Run generate script after schema changes

**Files that need manual updates:**
- `TECHNICAL_LOG.md` - Update when architecture changes
- `PROJECT_PLAN.md` - Update when completing/adding tasks
- `docs/TROUBLESHOOTING.md` - Add new errors as discovered
