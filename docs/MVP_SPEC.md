# MVP Spec — Organoid Media Recipe Predictor

> **For:** UI Developer
> **API Base URL:** TBD (will be provided once deployed)
> **Method:** `POST /predict`
> **Content-Type:** `application/json`

---

## Overview

A web form where a user enters clinical/genomic data about an organoid sample, submits it, and receives a predicted media recipe (which growth factors to include and at what concentrations).

---

## Input Fields

### Section 1: Required (1 field)

| Label | JSON Key | Type | UI Element | Options |
|-------|----------|------|-----------|---------|
| Cancer Type | `primary_site` | string | Dropdown | `Breast`, `Lung`, `Colon`, `Pancreas`, `Ovary`, `Stomach`, `Liver`, `Kidney`, `Prostate`, `Bladder`, `Esophagus`, `Head and Neck`, `Skin`, `Brain`, `Colorectal`, `Gastric`, `Other` |

> This is the **only required field**. All others are optional.

### Section 2: Clinical Info (7 fields, visible by default)

| Label | JSON Key | Type | UI Element | Options / Range |
|-------|----------|------|-----------|-----------------|
| Gender | `gender` | string | Dropdown | `Male`, `Female`, `Unknown` |
| Age at Diagnosis | `age_at_diagnosis_years` | number | Number input | 0–120 |
| Age at Acquisition | `age_at_acquisition_years` | number | Number input | 0–120 |
| Tissue Status | `tissue_status` | string | Dropdown | `Tumor`, `Normal`, `Metastatic`, `Unknown` |
| Disease Status | `disease_status` | string | Text input | Free text |
| Vital Status | `vital_status` | string | Dropdown | `Alive`, `Dead`, `Unknown` |
| Histological Grade | `histological_grade` | string | Dropdown | `G1`, `G2`, `G3`, `G4`, `Unknown` |

- If the user leaves a field blank, **omit it from the JSON** (don't send null).

### Section 3: Advanced — Genomic Data (collapsed by default)

This section should be behind a toggle/accordion labeled **"Advanced: Genomic Data"**.

Each field is a gene name with a VAF (Variant Allele Frequency) value.

| UI Behavior | Details |
|-------------|---------|
| Input type | Number (decimal) |
| Range | 0.00 – 1.00 |
| Placeholder | `0.00 – 1.00` |
| If blank | Omit from JSON (means "not sequenced") |

#### Top 10 Genes (show first, most impactful)

| Label | JSON Key |
|-------|----------|
| TP53 | `TP53_vaf` |
| KRAS | `KRAS_vaf` |
| APC | `APC_vaf` |
| PIK3CA | `PIK3CA_vaf` |
| ARID1A | `ARID1A_vaf` |
| KMT2D | `KMT2D_vaf` |
| RYR2 | `RYR2_vaf` |
| SYNE1 | `SYNE1_vaf` |
| TTN | `TTN_vaf` |
| MUC16 | `MUC16_vaf` |

#### Remaining 40 Genes (show below top 10, or in a sub-accordion)

| Label | JSON Key |
|-------|----------|
| FLG | `FLG_vaf` |
| OBSCN | `OBSCN_vaf` |
| CSMD3 | `CSMD3_vaf` |
| PCLO | `PCLO_vaf` |
| LRP1B | `LRP1B_vaf` |
| ZFHX4 | `ZFHX4_vaf` |
| CSMD1 | `CSMD1_vaf` |
| FAT4 | `FAT4_vaf` |
| FAT3 | `FAT3_vaf` |
| DNAH5 | `DNAH5_vaf` |
| FSIP2 | `FSIP2_vaf` |
| HYDIN | `HYDIN_vaf` |
| RYR1 | `RYR1_vaf` |
| HMCN1 | `HMCN1_vaf` |
| USH2A | `USH2A_vaf` |
| RYR3 | `RYR3_vaf` |
| APOB | `APOB_vaf` |
| AHNAK2 | `AHNAK2_vaf` |
| CCDC168 | `CCDC168_vaf` |
| ADGRV1 | `ADGRV1_vaf` |
| XIRP2 | `XIRP2_vaf` |
| DNAH11 | `DNAH11_vaf` |
| PLEC | `PLEC_vaf` |
| NEB | `NEB_vaf` |
| CSMD2 | `CSMD2_vaf` |
| RP1 | `RP1_vaf` |
| SPTA1 | `SPTA1_vaf` |
| MUC12 | `MUC12_vaf` |
| DCHS2 | `DCHS2_vaf` |
| EYS | `EYS_vaf` |
| DNAH3 | `DNAH3_vaf` |
| DNAH9 | `DNAH9_vaf` |
| MACF1 | `MACF1_vaf` |
| TNXB | `TNXB_vaf` |
| COL6A5 | `COL6A5_vaf` |
| DNAH7 | `DNAH7_vaf` |
| LRP2 | `LRP2_vaf` |
| PIEZO2 | `PIEZO2_vaf` |
| PKHD1L1 | `PKHD1L1_vaf` |
| KCNQ1 | `KCNQ1_vaf` |

---

## Sample Requests

### Minimal (just cancer type)

```json
POST /predict
Content-Type: application/json

{
  "primary_site": "Breast"
}
```

### With clinical info

```json
{
  "primary_site": "Colon",
  "gender": "Male",
  "age_at_diagnosis_years": 62,
  "tissue_status": "Tumor",
  "histological_grade": "G2"
}
```

### Full (clinical + genomic)

```json
{
  "primary_site": "Pancreas",
  "gender": "Female",
  "age_at_diagnosis_years": 55,
  "age_at_acquisition_years": 56,
  "tissue_status": "Tumor",
  "vital_status": "Alive",
  "histological_grade": "G3",
  "TP53_vaf": 0.45,
  "KRAS_vaf": 0.38,
  "PIK3CA_vaf": 0.0,
  "ARID1A_vaf": 0.12
}
```

---

## Response Format

### Success (200)

```json
{
  "predictions": {
    "egf": {
      "value": 50.0,
      "unit": "ng/mL",
      "type": "concentration",
      "action": "add",
      "status": "active"
    },
    "y27632": {
      "value": 1,
      "unit": "uM",
      "type": "binary",
      "action": "include",
      "status": "active"
    },
    "n_acetyl_cysteine": {
      "value": 1,
      "unit": "mM",
      "type": "binary",
      "action": "include",
      "status": "active"
    },
    "a83_01": {
      "value": 0,
      "unit": "nM",
      "type": "binary",
      "action": "exclude",
      "status": "active"
    },
    "sb202190": {
      "value": 0,
      "unit": "uM",
      "type": "binary",
      "action": "exclude",
      "status": "active"
    },
    "fgf2": {
      "value": 1,
      "unit": "ng/mL",
      "type": "binary",
      "action": "include",
      "status": "active"
    },
    "cholera_toxin": {
      "value": null,
      "unit": "ng/mL",
      "type": "concentration",
      "action": null,
      "status": "coming_soon",
      "status_label": "Coming Soon — insufficient training data (n=25)"
    },
    "insulin": {
      "value": null,
      "unit": "ug/mL",
      "type": "binary",
      "action": null,
      "status": "coming_soon",
      "status_label": "Coming Soon — insufficient training data (n=5)"
    },
    "wnt3a": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "r_spondin": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "noggin": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "chir99021": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "b27": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "n2": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "nicotinamide": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "gastrin": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "fgf7": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "fgf10": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "heparin": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "hydrocortisone": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "prostaglandin_e2": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "primocin": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "forskolin": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "heregulin": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    },
    "neuregulin": {
      "value": null,
      "unit": null,
      "type": null,
      "action": null,
      "status": "in_development",
      "status_label": "In Development — data collection in progress"
    }
  },
  "confidence": {
    "overall_grade": "B",
    "score": 0.78
  }
}
```

---

## Output Display Guide

### Factor Status Types

| `status` value | UI Display | Style |
|----------------|-----------|-------|
| `active` | Show prediction value + action | Normal row |
| `coming_soon` | Show "Coming Soon" badge | Greyed out, with tooltip showing `status_label` |
| `in_development` | Show "In Development" badge | Greyed out, with tooltip showing `status_label` |

### Display Rules

| Factor Field | How to Display |
|-------------|---------------|
| `status: "active"` + `type: "concentration"` | Show numeric value + unit (e.g., "EGF: 50 ng/mL") |
| `status: "active"` + `type: "binary"` | Show "Include" (value=1) or "Exclude" (value=0) |
| `status: "coming_soon"` | Show factor name + "Coming Soon" badge |
| `status: "in_development"` | Show factor name + "In Development" badge |
| `confidence.overall_grade` | Show as badge: A=green, B=blue, C=yellow, D=red |

### All 24 Factors (grouped for display)

#### Active Predictions (6 factors)

| Factor | Display Name | Prediction Type | Unit |
|--------|-------------|----------------|------|
| `egf` | EGF | Concentration | ng/mL |
| `y27632` | Y-27632 | Include/Exclude | uM |
| `n_acetyl_cysteine` | N-Acetylcysteine | Include/Exclude | mM |
| `a83_01` | A83-01 | Include/Exclude | nM |
| `sb202190` | SB202190 | Include/Exclude | uM |
| `fgf2` | FGF2 | Include/Exclude | ng/mL |

#### Coming Soon (2 factors — have some data, not enough to train)

| Factor | Display Name | Reason |
|--------|-------------|--------|
| `cholera_toxin` | Cholera Toxin | Insufficient training data (n=25) |
| `insulin` | Insulin | Insufficient training data (n=5) |

#### In Development (16 factors — no data yet, collection in progress)

| Factor | Display Name |
|--------|-------------|
| `wnt3a` | Wnt3a |
| `r_spondin` | R-spondin |
| `noggin` | Noggin |
| `chir99021` | CHIR99021 |
| `b27` | B27 Supplement |
| `n2` | N2 Supplement |
| `nicotinamide` | Nicotinamide |
| `gastrin` | Gastrin |
| `fgf7` | FGF7 |
| `fgf10` | FGF10 |
| `heparin` | Heparin |
| `hydrocortisone` | Hydrocortisone |
| `prostaglandin_e2` | Prostaglandin E2 |
| `primocin` | Primocin |
| `forskolin` | Forskolin |
| `heregulin` | Heregulin |
| `neuregulin` | Neuregulin |

---

## Error Responses

### Missing required field (400)

```json
{
  "error": "primary_site is required",
  "code": 400
}
```

### Invalid input (422)

```json
{
  "error": "Invalid primary_site: 'xyz'. Must be one of: Breast, Lung, Colon, ...",
  "code": 422
}
```

---

## UI Layout Reference

```
┌──────────────────────────────────────────────────┐
│  Organoid Media Recipe Predictor                 │
├──────────────────────────────────────────────────┤
│                                                  │
│  Cancer Type*:       [ Dropdown            ▼ ]   │
│                                                  │
│  ── Clinical Info (optional) ─────────────────── │
│  Gender:             [ Dropdown            ▼ ]   │
│  Age at Diagnosis:   [ _____ ]                   │
│  Age at Acquisition: [ _____ ]                   │
│  Tissue Status:      [ Dropdown            ▼ ]   │
│  Disease Status:     [ ____________________ ]    │
│  Vital Status:       [ Dropdown            ▼ ]   │
│  Histological Grade: [ Dropdown            ▼ ]   │
│                                                  │
│  ▶ Advanced: Genomic Data                        │
│  ┌──────────────────────────────────────────┐    │
│  │ TP53:   [____]    KRAS:   [____]        │    │
│  │ APC:    [____]    PIK3CA: [____]        │    │
│  │ ARID1A: [____]    KMT2D:  [____]        │    │
│  │ RYR2:   [____]    SYNE1:  [____]        │    │
│  │ TTN:    [____]    MUC16:  [____]        │    │
│  │ ── More genes ──                         │    │
│  │ FLG: [__] OBSCN: [__] CSMD3: [__] ...  │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│            [ Predict Media Recipe ]              │
│                                                  │
├──────────────────────────────────────────────────┤
│  PREDICTED MEDIA RECIPE             Grade: [B]   │
│  ─────────────────────────────────────────────   │
│                                                  │
│  ACTIVE PREDICTIONS                              │
│  EGF               50 ng/mL           Add        │
│  Y-27632           —                  Include     │
│  N-Acetylcysteine  —                  Include     │
│  A83-01            —                  Exclude     │
│  SB202190          —                  Exclude     │
│  FGF2              —                  Include     │
│                                                  │
│  COMING SOON                                     │
│  Cholera Toxin     ░░░░░░░░  Coming Soon         │
│  Insulin           ░░░░░░░░  Coming Soon         │
│                                                  │
│  IN DEVELOPMENT                                  │
│  Wnt3a             ░░░░░░░░  In Development      │
│  R-spondin         ░░░░░░░░  In Development      │
│  Noggin            ░░░░░░░░  In Development      │
│  CHIR99021         ░░░░░░░░  In Development      │
│  B27 Supplement    ░░░░░░░░  In Development      │
│  N2 Supplement     ░░░░░░░░  In Development      │
│  Nicotinamide      ░░░░░░░░  In Development      │
│  Gastrin           ░░░░░░░░  In Development      │
│  FGF7              ░░░░░░░░  In Development      │
│  FGF10             ░░░░░░░░  In Development      │
│  Heparin           ░░░░░░░░  In Development      │
│  Hydrocortisone    ░░░░░░░░  In Development      │
│  Prostaglandin E2  ░░░░░░░░  In Development      │
│  Primocin          ░░░░░░░░  In Development      │
│  Forskolin         ░░░░░░░░  In Development      │
│  Heregulin         ░░░░░░░░  In Development      │
│  Neuregulin        ░░░░░░░░  In Development      │
└──────────────────────────────────────────────────┘
```

---

## Summary

| | Count |
|---|---|
| Required inputs | 1 (Cancer Type) |
| Optional clinical inputs | 7 |
| Optional genomic inputs | 50 |
| **Total possible inputs** | **58** |
| Active output predictions | 6 media factors |
| Coming Soon factors | 2 |
| In Development factors | 16 |
| **Total output factors** | **24** |
