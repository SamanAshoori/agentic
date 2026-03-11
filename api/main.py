import json
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.state import session, STAGES
from api.schemas import (
    StatusResponse,
    UploadResponse,
    ETLConfirm,
    StatsConfirm,
    ModelConfirm,
    EvaluateConfirm,
    TargetRequest,
)
from fastapi.responses import FileResponse

from api.pipeline_runner import (
    get_dataset_summary,
    get_column_details,
    run_target_stats,
    apply_etl_decisions,
    run_stats,
    run_model,
    run_evaluate,
    run_scoring,
    run_descriptives,
    run_logistic_regression,
    generate_html_report,
    DATA,
)

app = FastAPI(title="Agentic ML Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status", response_model=StatusResponse)
def get_status():
    return StatusResponse(**session.to_dict())


@app.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    DATA.mkdir(exist_ok=True)
    dest = DATA / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    session.reset()
    session.dataset_path = str(dest)
    summary = get_dataset_summary(str(dest))
    session.dataset_summary = summary
    session.current_stage = "etl"
    return UploadResponse(**summary)


@app.get("/columns")
def get_columns():
    """Detailed per-column stats for the uploaded dataset."""
    if not session.dataset_path:
        raise HTTPException(400, "No dataset uploaded")
    return get_column_details(session.dataset_path)


@app.post("/analyze/target")
def analyze_target(payload: TargetRequest):
    """Run statistical tests for all columns against the selected target."""
    if not session.dataset_path:
        raise HTTPException(400, "No dataset uploaded")
    try:
        results = run_target_stats(session.dataset_path, payload.target)
        session.stage_status["etl"] = "awaiting_review"
        session.current_stage = "etl"
        return results
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# ETL — user-driven column selection
# ---------------------------------------------------------------------------

@app.post("/confirm/etl")
def confirm_etl(payload: ETLConfirm):
    if not session.dataset_path:
        raise HTTPException(400, "No dataset uploaded")
    try:
        result = apply_etl_decisions(session.dataset_path, payload)
        session.confirmed_outputs["etl"] = {
            "target": payload.target,
            "columns_kept": result["columns_kept"],
            "columns_dropped": result["columns_dropped"],
            "cleaned_shape": result["cleaned_shape"],
            "cleaned_path": result["cleaned_path"],
        }
        session.stage_status["etl"] = "confirmed"
        return {"status": "confirmed", "stage": "etl", **result}
    except Exception as e:
        raise HTTPException(500, f"ETL failed: {str(e)}")


# ---------------------------------------------------------------------------
# Stats — direct computation
# ---------------------------------------------------------------------------

@app.post("/run/stats")
def run_stats_stage():
    if session.stage_status["etl"] not in ("confirmed", "complete"):
        raise HTTPException(400, "ETL stage must be confirmed first")
    target = session.confirmed_outputs.get("etl", {}).get("target")
    cleaned_path = session.confirmed_outputs.get("etl", {}).get("cleaned_path")
    if not target:
        raise HTTPException(400, "No target variable set")
    if not cleaned_path:
        raise HTTPException(400, "Cleaned dataset path not found — re-confirm ETL")
    try:
        session.stage_status["stats"] = "running"
        result = run_stats(target, cleaned_path)
        session.stage_outputs["stats"] = result
        session.stage_status["stats"] = "awaiting_review"
        session.current_stage = "stats"
        return result
    except Exception as e:
        session.stage_status["stats"] = "pending"
        raise HTTPException(500, f"Stats failed: {str(e)}")


@app.post("/confirm/stats")
def confirm_stats(payload: StatsConfirm):
    if session.stage_status["stats"] != "awaiting_review":
        raise HTTPException(400, "Stats is not awaiting review")
    session.confirmed_outputs["stats"] = payload.model_dump()
    session.stage_status["stats"] = "confirmed"
    return {"status": "confirmed", "stage": "stats"}


# ---------------------------------------------------------------------------
# Model — direct computation
# ---------------------------------------------------------------------------

@app.post("/run/model")
def run_model_stage(payload: ModelConfirm):
    if session.stage_status["stats"] not in ("confirmed", "complete"):
        raise HTTPException(400, "Stats stage must be confirmed first")
    target = session.confirmed_outputs["etl"]["target"]
    features = session.confirmed_outputs["stats"]["selected_features"]
    cleaned_path = session.confirmed_outputs["etl"]["cleaned_path"]
    hp = payload.hyperparameters
    try:
        session.stage_status["model"] = "running"
        result = run_model(
            target=target,
            selected_features=features,
            cleaned_path=cleaned_path,
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", 10),
            class_weight_mode=hp.get("class_weight", "balanced"),
            test_split=hp.get("test_split", 0.2),
        )
        session.stage_outputs["model"] = result
        session.stage_status["model"] = "awaiting_review"
        session.current_stage = "model"
        return result
    except Exception as e:
        session.stage_status["model"] = "pending"
        raise HTTPException(500, f"Model training failed: {str(e)}")


@app.post("/confirm/model")
def confirm_model(payload: ModelConfirm):
    if session.stage_status["model"] != "awaiting_review":
        raise HTTPException(400, "Model is not awaiting review")
    session.confirmed_outputs["model"] = payload.model_dump()
    session.stage_status["model"] = "confirmed"
    return {"status": "confirmed", "stage": "model"}


# ---------------------------------------------------------------------------
# Evaluate — direct computation
# ---------------------------------------------------------------------------

@app.post("/run/evaluate")
def run_evaluate_stage():
    if session.stage_status["model"] not in ("confirmed", "complete"):
        raise HTTPException(400, "Model stage must be confirmed first")
    target = session.confirmed_outputs["etl"]["target"]
    features = session.confirmed_outputs["stats"]["selected_features"]
    cleaned_path = session.confirmed_outputs["etl"]["cleaned_path"]
    try:
        session.stage_status["evaluate"] = "running"
        result = run_evaluate(target=target, selected_features=features, cleaned_path=cleaned_path)
        session.stage_outputs["evaluate"] = result
        session.stage_status["evaluate"] = "awaiting_review"
        session.current_stage = "evaluate"
        return result
    except Exception as e:
        session.stage_status["evaluate"] = "pending"
        raise HTTPException(500, f"Evaluation failed: {str(e)}")


@app.post("/confirm/evaluate")
def confirm_evaluate(payload: EvaluateConfirm):
    if session.stage_status["evaluate"] != "awaiting_review":
        raise HTTPException(400, "Evaluate is not awaiting review")
    session.confirmed_outputs["evaluate"] = payload.model_dump()
    session.stage_status["evaluate"] = "confirmed"
    return {"status": "confirmed", "stage": "evaluate"}


# ---------------------------------------------------------------------------
# Scoring — probability output with segments
# ---------------------------------------------------------------------------

@app.post("/run/scoring")
def run_scoring_stage():
    if session.stage_status["evaluate"] not in ("confirmed", "complete"):
        raise HTTPException(400, "Evaluate stage must be confirmed first")
    target = session.confirmed_outputs["etl"]["target"]
    features = session.confirmed_outputs["stats"]["selected_features"]
    cleaned_path = session.confirmed_outputs["etl"]["cleaned_path"]
    try:
        session.stage_status["scoring"] = "running"
        result = run_scoring(target=target, selected_features=features, cleaned_path=cleaned_path)
        session.stage_outputs["scoring"] = result
        session.stage_status["scoring"] = "complete"
        session.current_stage = "complete"
        return result
    except Exception as e:
        session.stage_status["scoring"] = "pending"
        raise HTTPException(500, f"Scoring failed: {str(e)}")


@app.get("/download/scored")
def download_scored():
    path = DATA / "scored_output.csv"
    if not path.exists():
        raise HTTPException(404, "Scored output not found")
    return FileResponse(path, media_type="text/csv", filename="scored_output.csv")


# ---------------------------------------------------------------------------
# Descriptives — SPSS-style class comparison
# ---------------------------------------------------------------------------

@app.post("/run/descriptives")
def run_descriptives_stage(payload: TargetRequest):
    if not session.dataset_path:
        raise HTTPException(400, "No dataset uploaded")
    try:
        result = run_descriptives(session.dataset_path, payload.target)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

@app.post("/run/logistic")
def run_logistic_stage(payload: ModelConfirm):
    if session.stage_status["stats"] not in ("confirmed", "complete"):
        raise HTTPException(400, "Stats stage must be confirmed first")
    target = session.confirmed_outputs["etl"]["target"]
    features = session.confirmed_outputs["stats"]["selected_features"]
    cleaned_path = session.confirmed_outputs["etl"]["cleaned_path"]
    hp = payload.hyperparameters
    try:
        session.stage_status["model"] = "running"
        result = run_logistic_regression(
            target=target,
            selected_features=features,
            cleaned_path=cleaned_path,
            test_split=hp.get("test_split", 0.2),
            max_iter=hp.get("max_iter", 1000),
        )
        session.stage_outputs["model"] = result
        session.stage_status["model"] = "awaiting_review"
        session.current_stage = "model"
        return result
    except Exception as e:
        session.stage_status["model"] = "pending"
        raise HTTPException(500, f"Logistic regression failed: {str(e)}")


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

@app.get("/download/report")
def download_report():
    report_data = {
        "session": session.to_dict(),
        "stage_outputs": {k: v for k, v in session.stage_outputs.items()},
    }
    for fname in ["model_metrics", "eval_report", "selected_features"]:
        path = DATA / f"{fname}.json"
        if path.exists():
            with open(path) as f:
                report_data[fname] = json.load(f)

    html = generate_html_report(report_data)
    report_path = DATA / "pipeline_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    return FileResponse(report_path, media_type="text/html", filename="ml_pipeline_report.html")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

@app.get("/output/{stage}")
def get_output(stage: str):
    if stage not in STAGES:
        raise HTTPException(400, f"Unknown stage: {stage}")
    if stage not in session.stage_outputs:
        raise HTTPException(404, f"No output for stage {stage}")
    return {"stage": stage, "status": session.stage_status[stage], "data": session.stage_outputs[stage]}


@app.post("/reset")
def reset_session():
    session.reset()
    return {"status": "reset"}
