# Troubleshooting Guide

## Path Errors

### Error: `FileNotFoundError: database/organoid_data.db`
**Cause:** Database not in expected location
**Fix:**
1. Verify database exists: `ls -la database/`
2. If missing, restore from backup or re-download
3. Check config/paths.py for correct path

### Error: `ModuleNotFoundError: No module named 'config'`
**Cause:** Running script from wrong directory or package not installed
**Fix:**
```bash
cd ~/Projects/organoid-media-ml
pip install -e .  # Install package in editable mode
```

### Error: `ModuleNotFoundError: No module named 'src'`
**Cause:** Package not installed or wrong working directory
**Fix:**
```bash
cd ~/Projects/organoid-media-ml
pip install -e .
# Or add to path manually:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Database Errors

### Error: `sqlite3.OperationalError: database is locked`
**Cause:** Another process has the database open
**Fix:**
1. Close other Python processes or notebooks
2. Check for zombie processes: `lsof database/organoid_data.db`
3. If stuck: `kill -9 <PID>`

### Error: `no such table: master_dataset_v2`
**Cause:** Database schema incomplete
**Fix:**
```bash
python scripts/data_pipeline/organoid_db_cleanup.py
```

### Error: `sqlite3.OperationalError: unable to open database file`
**Cause:** Path incorrect or permissions issue
**Fix:**
```bash
# Check path exists
ls -la ~/Projects/organoid-media-ml/database/

# Check permissions
chmod 644 database/organoid_data.db
```

---

## Environment Errors

### Error: `ImportError: No module named 'sklearn'`
**Cause:** Dependencies not installed
**Fix:**
```bash
pip install -r requirements.txt
```

### Error: Version mismatch warnings
**Cause:** Different sklearn version than model was trained with
**Fix:**
```bash
pip install scikit-learn==1.3.0  # Match trained version
```

### Error: `ImportError: cannot import name 'MediaRecipeGenerator'`
**Cause:** Import path changed or module not found
**Fix:**
```python
# Correct import:
from src.models import MediaRecipeGenerator
# Or:
from src.models.media_recipe_generator import MediaRecipeGenerator
```

---

## Training Errors

### Error: `ValueError: Found input variables with inconsistent numbers of samples`
**Cause:** X and y have different lengths, usually due to masking mismatch
**Fix:**
```python
# Ensure mask is applied to both X and y
mask = y.notna()
X_masked = X[mask]
y_masked = y[mask]
```

### Error: `ValueError: y contains previously unseen labels`
**Cause:** Test set has labels not in training set
**Fix:**
- Check for data leakage in train/test split
- Use stratified splitting for classification tasks

### Error: `MemoryError` during training
**Cause:** Dataset too large for available memory
**Fix:**
```python
# Reduce dataset size or use incremental learning
# Or increase available memory
```

---

## Model Loading Errors

### Error: `FileNotFoundError: media_recipe_generator.joblib`
**Cause:** Model not trained yet or wrong path
**Fix:**
```bash
# Check if model exists
ls -la model_artifacts/

# If missing, retrain:
python train.py
```

### Error: `sklearn version mismatch` warning
**Cause:** Model trained with different sklearn version
**Fix:**
```bash
# Option 1: Match the version
pip install scikit-learn==<version_used_for_training>

# Option 2: Retrain with current version
python train.py
```

---

## Common Mistakes

### 1. Running scripts from wrong directory
Always run from project root:
```bash
cd ~/Projects/organoid-media-ml
python train.py  # Correct
```

### 2. Forgetting to activate virtual environment
```bash
source venv/bin/activate  # or conda activate <env_name>
```

### 3. Using relative imports incorrectly
```python
# Wrong (if running as script):
from .models import MediaRecipeGenerator

# Correct:
from src.models import MediaRecipeGenerator
```

### 4. Hardcoding paths
```python
# Wrong:
DB_PATH = "/Users/fahd838/Desktop/organoid_data.db"

# Correct:
from config.paths import DB_PATH
```

---

## Getting Help

If you encounter an error not listed here:
1. Check the full traceback for clues
2. Search the codebase for similar patterns
3. Check if the database connection works:
   ```bash
   python scripts/verification/verify_db_link.py
   ```
4. Add the error and solution to this file for future reference
