from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json

from core.ingestion import IngestionEngine
from core.profiling import profile_dataset
from core.domain import detect_domain
from core.cleaning import CleaningEngine
from core.statistics_engine import descriptive_stats, correlation_matrix
from core.analytics import AnalyticsEngine
from core.ml_engine import MLEngine, detect_task
from core.visualization import ChartFactory

app = FastAPI(title="Analyze API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analytics = AnalyticsEngine()
ml = MLEngine()

# In-memory storage for uploaded datasets (for development purposes)
# In production, this should be a database or cloud storage
DATASETS = {}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        buf = io.BytesIO(content)
        buf.name = file.filename
        
        engine = IngestionEngine()
        df, report = engine.parse_file(buf)
        
        dataset_id = file.filename # simplified for now
        DATASETS[dataset_id] = df
        
        profile = profile_dataset(df)
        domain = detect_domain(df)
        
        schema_info = []
        for name, p in profile.schema.items():
            schema_info.append({
                "column": name,
                "type": p.inferred_type.value,
                "dtype": str(p.dtype),
                "unique": p.n_unique,
                "missing_pct": p.missing_pct
            })
            
        return {
            "dataset_id": dataset_id,
            "rows": profile.shape[0],
            "columns": profile.shape[1],
            "memory": profile.memory_human,
            "quality_score": profile.quality.overall,
            "domain": domain.domain,
            "domain_confidence": domain.confidence,
            "schema": schema_info,
            "warnings": report.warnings if report else []
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dataset/{dataset_id}/preview")
async def preview_dataset(dataset_id: str, limit: int = 100):
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    df = DATASETS[dataset_id]
    preview = df.head(limit).to_dict(orient="records")
    return {"data": preview}

@app.post("/dataset/{dataset_id}/clean")
async def clean_dataset(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    try:
        df = DATASETS[dataset_id]
        clean_df, report = CleaningEngine().run(df)
        
        # Overwrite or store as cleaned
        DATASETS[f"{dataset_id}_cleaned"] = clean_df
        
        return {
            "cleaned_dataset_id": f"{dataset_id}_cleaned",
            "rows_before": report.rows_before,
            "rows_after": report.rows_after,
            "cols_before": report.cols_before,
            "cols_after": report.cols_after,
            "duplicates_removed": report.duplicate_rows_removed,
            "dropped_columns": report.dropped_columns,
            "audit_log": report.audit_log
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dataset/{dataset_id}/stats")
async def get_stats(dataset_id: str):
    if dataset_id not in DATASETS:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    df = DATASETS[dataset_id]
    desc = descriptive_stats(df)
    
    # NaN and Inf are not JSON serializable, so we replace them
    desc_dict = desc.replace({pd.NA: None, float('nan'): None}).to_dict(orient="index")
    
    return {"stats": desc_dict}

from mangum import Mangum

handler = Mangum(app)