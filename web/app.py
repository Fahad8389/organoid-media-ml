"""
FastAPI Web Server for Organoid Media Recipe Predictor

This is an early version - predictions are for research purposes only.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn

from beta.api import BetaAPI

# =============================================================================
# App Configuration
# =============================================================================

APP_VERSION = "0.1.0-beta"
APP_NOTICE = "This is an early version — predictions are for research purposes only."

SUPPORT_INFO = {
    "email": "F.A.B.Alotaibi@gmail.com",
    "message": "For support and feedback, please contact us."
}

# Valid cancer types
VALID_CANCER_TYPES = [
    "Breast", "Lung", "Colon", "Pancreas", "Ovary", "Stomach", "Liver",
    "Kidney", "Prostate", "Bladder", "Esophagus", "Head and Neck", "Skin",
    "Brain", "Colorectal", "Gastric", "Other"
]

# Factor metadata
FACTOR_METADATA = {
    # Active factors (6)
    "egf": {"display_name": "EGF", "unit": "ng/mL", "type": "concentration", "status": "active"},
    "y27632": {"display_name": "Y-27632", "unit": "uM", "type": "binary", "status": "active"},
    "n_acetyl_cysteine": {"display_name": "N-Acetylcysteine", "unit": "mM", "type": "binary", "status": "active"},
    "a83_01": {"display_name": "A83-01", "unit": "nM", "type": "binary", "status": "active"},
    "sb202190": {"display_name": "SB202190", "unit": "uM", "type": "binary", "status": "active"},
    "fgf2": {"display_name": "FGF2", "unit": "ng/mL", "type": "binary", "status": "active"},
    # Coming soon factors (2)
    "cholera_toxin": {"display_name": "Cholera Toxin", "unit": "ng/mL", "type": "concentration", "status": "coming_soon", "status_label": "Coming Soon — insufficient training data (n=25)"},
    "insulin": {"display_name": "Insulin", "unit": "ug/mL", "type": "binary", "status": "coming_soon", "status_label": "Coming Soon — insufficient training data (n=5)"},
    # In development factors (16)
    "wnt3a": {"display_name": "Wnt3a", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "r_spondin": {"display_name": "R-spondin", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "noggin": {"display_name": "Noggin", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "chir99021": {"display_name": "CHIR99021", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "b27": {"display_name": "B27 Supplement", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "n2": {"display_name": "N2 Supplement", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "nicotinamide": {"display_name": "Nicotinamide", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "gastrin": {"display_name": "Gastrin", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "fgf7": {"display_name": "FGF7", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "fgf10": {"display_name": "FGF10", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "heparin": {"display_name": "Heparin", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "hydrocortisone": {"display_name": "Hydrocortisone", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "prostaglandin_e2": {"display_name": "Prostaglandin E2", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "primocin": {"display_name": "Primocin", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "forskolin": {"display_name": "Forskolin", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "heregulin": {"display_name": "Heregulin", "status": "in_development", "status_label": "In Development — data collection in progress"},
    "neuregulin": {"display_name": "Neuregulin", "status": "in_development", "status_label": "In Development — data collection in progress"},
}

# =============================================================================
# Pydantic Models
# =============================================================================

class PredictionRequest(BaseModel):
    """Input request for media recipe prediction."""
    # Required
    primary_site: str = Field(..., description="Cancer type (required)")

    # Clinical info (optional)
    gender: Optional[str] = Field(None, description="Male, Female, or Unknown")
    age_at_diagnosis_years: Optional[int] = Field(None, ge=0, le=120)
    age_at_acquisition_years: Optional[int] = Field(None, ge=0, le=120)
    tissue_status: Optional[str] = Field(None, description="Tumor, Normal, Metastatic, or Unknown")
    disease_status: Optional[str] = Field(None)
    vital_status: Optional[str] = Field(None, description="Alive, Dead, or Unknown")
    histological_grade: Optional[str] = Field(None, description="G1, G2, G3, G4, or Unknown")

    # Genomic data (optional) - Top 50 genes
    TP53_vaf: Optional[float] = Field(None, ge=0, le=1)
    KRAS_vaf: Optional[float] = Field(None, ge=0, le=1)
    APC_vaf: Optional[float] = Field(None, ge=0, le=1)
    PIK3CA_vaf: Optional[float] = Field(None, ge=0, le=1)
    ARID1A_vaf: Optional[float] = Field(None, ge=0, le=1)
    KMT2D_vaf: Optional[float] = Field(None, ge=0, le=1)
    RYR2_vaf: Optional[float] = Field(None, ge=0, le=1)
    SYNE1_vaf: Optional[float] = Field(None, ge=0, le=1)
    TTN_vaf: Optional[float] = Field(None, ge=0, le=1)
    MUC16_vaf: Optional[float] = Field(None, ge=0, le=1)
    FLG_vaf: Optional[float] = Field(None, ge=0, le=1)
    OBSCN_vaf: Optional[float] = Field(None, ge=0, le=1)
    CSMD3_vaf: Optional[float] = Field(None, ge=0, le=1)
    PCLO_vaf: Optional[float] = Field(None, ge=0, le=1)
    LRP1B_vaf: Optional[float] = Field(None, ge=0, le=1)
    ZFHX4_vaf: Optional[float] = Field(None, ge=0, le=1)
    CSMD1_vaf: Optional[float] = Field(None, ge=0, le=1)
    FAT4_vaf: Optional[float] = Field(None, ge=0, le=1)
    FAT3_vaf: Optional[float] = Field(None, ge=0, le=1)
    DNAH5_vaf: Optional[float] = Field(None, ge=0, le=1)
    FSIP2_vaf: Optional[float] = Field(None, ge=0, le=1)
    HYDIN_vaf: Optional[float] = Field(None, ge=0, le=1)
    RYR1_vaf: Optional[float] = Field(None, ge=0, le=1)
    HMCN1_vaf: Optional[float] = Field(None, ge=0, le=1)
    USH2A_vaf: Optional[float] = Field(None, ge=0, le=1)
    RYR3_vaf: Optional[float] = Field(None, ge=0, le=1)
    APOB_vaf: Optional[float] = Field(None, ge=0, le=1)
    AHNAK2_vaf: Optional[float] = Field(None, ge=0, le=1)
    CCDC168_vaf: Optional[float] = Field(None, ge=0, le=1)
    ADGRV1_vaf: Optional[float] = Field(None, ge=0, le=1)
    XIRP2_vaf: Optional[float] = Field(None, ge=0, le=1)
    DNAH11_vaf: Optional[float] = Field(None, ge=0, le=1)
    PLEC_vaf: Optional[float] = Field(None, ge=0, le=1)
    NEB_vaf: Optional[float] = Field(None, ge=0, le=1)
    CSMD2_vaf: Optional[float] = Field(None, ge=0, le=1)
    RP1_vaf: Optional[float] = Field(None, ge=0, le=1)
    SPTA1_vaf: Optional[float] = Field(None, ge=0, le=1)
    MUC12_vaf: Optional[float] = Field(None, ge=0, le=1)
    DCHS2_vaf: Optional[float] = Field(None, ge=0, le=1)
    EYS_vaf: Optional[float] = Field(None, ge=0, le=1)
    DNAH3_vaf: Optional[float] = Field(None, ge=0, le=1)
    DNAH9_vaf: Optional[float] = Field(None, ge=0, le=1)
    MACF1_vaf: Optional[float] = Field(None, ge=0, le=1)
    TNXB_vaf: Optional[float] = Field(None, ge=0, le=1)
    COL6A5_vaf: Optional[float] = Field(None, ge=0, le=1)
    DNAH7_vaf: Optional[float] = Field(None, ge=0, le=1)
    LRP2_vaf: Optional[float] = Field(None, ge=0, le=1)
    PIEZO2_vaf: Optional[float] = Field(None, ge=0, le=1)
    PKHD1L1_vaf: Optional[float] = Field(None, ge=0, le=1)
    KCNQ1_vaf: Optional[float] = Field(None, ge=0, le=1)


class FactorPrediction(BaseModel):
    """Single factor prediction."""
    value: Optional[float] = None
    unit: Optional[str] = None
    type: Optional[str] = None
    action: Optional[str] = None
    status: str
    status_label: Optional[str] = None
    grade: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response from media recipe prediction."""
    predictions: Dict[str, FactorPrediction]
    confidence: Dict[str, Any]
    version: str
    notice: str
    support: Dict[str, str]


# =============================================================================
# Static Files
# =============================================================================

STATIC_DIR = Path(__file__).parent / "static"

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Organoid Media Recipe Predictor",
    description="ML-powered prediction of optimal culture media formulations for organoid models.",
    version=APP_VERSION,
)

# CORS - allow all origins for now (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load model on startup
api: Optional[BetaAPI] = None


@app.on_event("startup")
async def load_model():
    """Load the ML model on startup."""
    global api
    model_dir = PROJECT_ROOT / "beta_output"
    try:
        api = BetaAPI.load(str(model_dir))
        print(f"Model loaded from {model_dir}")
    except Exception as e:
        print(f"WARNING: Could not load model: {e}")
        print("API will return mock responses until model is available.")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Serve the frontend HTML."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api")
async def api_info():
    """API info endpoint."""
    return {
        "name": "Organoid Media Recipe Predictor API",
        "version": APP_VERSION,
        "notice": APP_NOTICE,
        "support": SUPPORT_INFO,
        "endpoints": {
            "/predict": "POST - Predict media recipe",
            "/health": "GET - Health check",
            "/factors": "GET - List all media factors",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": api is not None,
        "version": APP_VERSION
    }


@app.get("/factors")
async def list_factors():
    """List all media factors with their status."""
    return {
        "factors": FACTOR_METADATA,
        "counts": {
            "active": sum(1 for f in FACTOR_METADATA.values() if f.get("status") == "active"),
            "coming_soon": sum(1 for f in FACTOR_METADATA.values() if f.get("status") == "coming_soon"),
            "in_development": sum(1 for f in FACTOR_METADATA.values() if f.get("status") == "in_development"),
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict optimal media recipe for an organoid sample.

    Only `primary_site` (cancer type) is required. All other fields are optional
    but will improve prediction accuracy if provided.
    """
    # Validate cancer type
    if request.primary_site not in VALID_CANCER_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid primary_site: '{request.primary_site}'. Must be one of: {', '.join(VALID_CANCER_TYPES)}"
        )

    # Check if model is loaded
    if api is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )

    # Build sample dict with defaults for required preprocessor columns
    raw_sample = request.model_dump()

    # All 50 VAF genes expected by the model
    VAF_GENES = [
        'TP53', 'KRAS', 'APC', 'PIK3CA', 'ARID1A', 'KMT2D', 'RYR2', 'SYNE1', 'TTN', 'MUC16',
        'FLG', 'OBSCN', 'CSMD3', 'PCLO', 'LRP1B', 'ZFHX4', 'CSMD1', 'FAT4', 'FAT3', 'DNAH5',
        'FSIP2', 'HYDIN', 'RYR1', 'HMCN1', 'USH2A', 'RYR3', 'APOB', 'AHNAK2', 'CCDC168', 'ADGRV1',
        'XIRP2', 'DNAH11', 'PLEC', 'NEB', 'CSMD2', 'RP1', 'SPTA1', 'MUC12', 'DCHS2', 'EYS',
        'DNAH3', 'DNAH9', 'MACF1', 'TNXB', 'COL6A5', 'DNAH7', 'LRP2', 'PIEZO2', 'PKHD1L1', 'KCNQ1'
    ]

    # Default values for categorical columns the preprocessor expects
    defaults = {
        'gender': 'Unknown',
        'tissue_status': 'Unknown',
        'disease_status': 'Unknown',
        'vital_status': 'Unknown',
        'histological_grade': 'Unknown',
        'model_type': 'Unknown',
        'age_at_acquisition_years': None,
        'age_at_diagnosis_years': None,
    }

    # Add all VAF columns with None (will be treated as "not sequenced")
    for gene in VAF_GENES:
        defaults[f'{gene}_vaf'] = None

    # Start with defaults, then overlay user-provided values
    sample = defaults.copy()
    for k, v in raw_sample.items():
        if v is not None:
            sample[k] = v

    # Get prediction
    result = api.predict_single(sample)

    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=result.error
        )

    # Build response with all 24 factors
    predictions = {}

    for factor_key, metadata in FACTOR_METADATA.items():
        status = metadata.get("status", "in_development")

        if status == "active" and factor_key in result.recipe:
            # Active factor with prediction
            raw_value = result.recipe.get(factor_key)
            factor_conf = result.confidence.get(factor_key, 0) if result.confidence else 0

            # Determine action and value based on type
            factor_type = metadata.get("type", "binary")

            if factor_type == "concentration":
                # Regression - show actual value
                value = raw_value
                action = "add" if value and value > 0 else None
            else:
                # Binary - include/exclude
                if raw_value is not None and raw_value > 0.5:
                    value = 1
                    action = "include"
                else:
                    value = 0
                    action = "exclude"

            # Assign grade based on confidence
            if factor_conf >= 0.9:
                grade = "A"
            elif factor_conf >= 0.75:
                grade = "B"
            elif factor_conf >= 0.5:
                grade = "C"
            else:
                grade = "D"

            predictions[factor_key] = FactorPrediction(
                value=value,
                unit=metadata.get("unit"),
                type=factor_type,
                action=action,
                status="active",
                grade=grade
            )
        else:
            # Coming soon or in development
            predictions[factor_key] = FactorPrediction(
                value=None,
                unit=metadata.get("unit"),
                type=metadata.get("type"),
                action=None,
                status=status,
                status_label=metadata.get("status_label")
            )

    return PredictionResponse(
        predictions=predictions,
        confidence={
            "overall_grade": result.grade,
            "score": result.overall_confidence
        },
        version=APP_VERSION,
        notice=APP_NOTICE,
        support=SUPPORT_INFO
    )


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
