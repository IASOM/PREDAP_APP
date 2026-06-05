from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
import sys
import os
from uuid import uuid4
import json
from datetime import datetime, timezone
import pandas as pd
import redis
from redis.exceptions import RedisError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.base_transformer_config import BaseTransformerConfig

from production.add_new_data_pipeline import AddNewDataPipeline
from production.model_reconstruction_pipeline import ModelPredictionPipeline
from api.schemas.production_schemas import AddNewDataRequest
from api.schemas.production_schemas import ModelReconstructionRequest



router = APIRouter(prefix="/production", tags=["production"])

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()

def _job_key(job_id: str) -> str:
    return f"jobs:model_reconstruction:{job_id}"

def _save_job(job_id: str, payload: dict) -> None:
    key = _job_key(job_id)
    redis_client.set(key, json.dumps(payload), ex=JOB_TTL_SECONDS)

def _read_job(job_id: str):
    raw = redis_client.get(_job_key(job_id))
    return json.loads(raw) if raw else None

def _set_job_status(job_id: str, status: str, updates: Optional[dict] = None):
    data = _read_job(job_id) or {}
    data["job_id"] = job_id
    data["status"] = status
    data["updated_at"] = _utc_now()
    if updates:
        data.update(updates)
    _save_job(job_id, data)


@router.post("/add_new_data")
def add_new_data(request: AddNewDataRequest = None):
    """
    Appends a new row of data to the dataset. If manual data isn't provided,
    it imputes values using a 3-year seasonal mean (same day/month).
    """
    try:
        config = BaseTransformerConfig()
        pipeline = AddNewDataPipeline(config)

        # Use request values or fall back to config defaults
        new_data_path = request.new_data_path if request and request.new_data_path else config.data_path
        cutoff_date = request.cutoff_date if request and request.cutoff_date else config.cutoff_date
        max_date = request.max_date if request and request.max_date else config.max_date
        eliminate_covid_data = request.eliminate_covid_data if request and request.eliminate_covid_data is not None else config.eliminate_covid_data
        covid_token = request.covid_token if request and request.covid_token is not None else config.covid_token
        provided_data = request.provided_data if request else None
        save_path = request.save_path if request and request.save_path else "../data/FINAL_DB1"
        delete_old = request.delete_old if request and request.delete_old is not None else True

        # Run the pipeline to add new data
        updated_df = pipeline.add_new_data(
            new_data_path=new_data_path,
            cutoff_date=cutoff_date,
            max_date=max_date,
            eliminate_covid_data=eliminate_covid_data,
            covid_token=covid_token,
            provided_data=provided_data
        )

        # Generate save name from the data path
        save_name = new_data_path.split('/')[-1].replace('.parquet', '')

        # Save the updated data
        saved_path = pipeline.save_updated_data(
            df=updated_df,
            save_path=save_path,
            save_name=save_name,
            delete_old=delete_old
        )

        # Get summary info from the updated dataframe
        last_row = updated_df.iloc[-1].to_dict()
        last_row['timestamp'] = str(last_row['timestamp'])

        return JSONResponse(content={
            "status": "success",
            "message": "New data added successfully!",
            "saved_path": saved_path,
            "total_rows": len(updated_df),
            "new_row": last_row
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Data file not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding new data: {str(e)}")


def run_model_reconstruction_job(job_id: str, payload: dict) -> None:
    try:
        _set_job_status(job_id, "running")
        config = BaseTransformerConfig(
            code=payload["code"],
            head_size=payload["head_size"],
            num_heads=payload["num_heads"],
            ff_dim=payload["ff_dim"],
            num_transformer_blocks=payload["num_transformer_blocks"],
            mlp_units=payload["mlp_units"],
            activation_function=payload["activation_function"],
            dropout=payload["dropout"],
            learning_rate=payload["learning_rate"],
            epochs=payload["epochs"],
            batch_size=payload["batch_size"],
            cutoff_date=payload["cutoff_date"],
            covid_token=payload["covid_token"],
            positional_encoding=payload["positional_encoding"],
            evaluate_model=payload["evaluate_model"],
            data_path=payload["data_path"],
            model_folder=payload.get("model_folder") or BaseTransformerConfig.model_folder,
        )
        pipeline = ModelPredictionPipeline(config)
        input_directory = payload["data_path"]
        old_input_directory = payload.get("old_data_path") or input_directory

        final_output_df = pipeline.run_reconstruct_save_results_pipeline(
            input_directory=input_directory,
            old_input_directory=old_input_directory,
            code=payload["code"],
            LOOKBACK_LIST=payload["lookback_list"],
            FORECAST_LIST=payload["forecast_horizon_list"],
            final_output_predictions=None,
            final_output_df=pd.DataFrame(),
            prediction_dates=payload.get("prediction_dates"),
        )

        output_path = payload.get("save_path") or "../production_predictions/final_output_predictions"
        pipeline.save_final_output_predictions(
            final_output_df,
            output_path=output_path,
        )

        _set_job_status(
                job_id,
                "succeeded",
                {
                    "finished_at": _utc_now(),
                    "result": {
                        "rows": int(len(final_output_df)),
                        "output_path": output_path,
                    },
                    "error": None,
                },
            )

    except Exception as e:
        _set_job_status(
            job_id,
            "failed",
            {
                "finished_at": _utc_now(),
                "error": str(e),
            },
        )


@router.post("/model_reconstruction_pipeline", status_code=202)
def model_reconstruction_pipeline(
    request: ModelReconstructionRequest,
    background_tasks: BackgroundTasks,
):
    job_id = str(uuid4())
    payload = request.model_dump()

    try:
        _save_job(
            job_id,
            {
                "job_id": job_id,
                "status": "queued",
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "error": None,
                "result": None,
            },
        )
    except RedisError as e:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {str(e)}")
    background_tasks.add_task(run_model_reconstruction_job, job_id, payload)
    return JSONResponse(
        content={
            "status": "queued",
            "job_id": job_id,
            "status_endpoint": f"/production/model_reconstruction_pipeline/{job_id}",
            "message": "Model reconstruction started in the background.",
        },
        status_code=202,
    )

@router.get("/model_reconstruction_pipeline/{job_id}")
def model_reconstruction_pipeline_status(job_id: str):
    try:
        job = _read_job(job_id)
    except RedisError as e:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {str(e)}")

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(content=job)

'''@router.post("/model_reconstruction_pipeline")
def model_reconstruction_pipeline(request: ModelReconstructionRequest):
    """
    Endpoint to trigger the model reconstruction pipeline. This will retrain the model using the updated dataset.
    """

    try:
        # Placeholder for actual model reconstruction logic
        # You would call your model training functions here, passing in the updated dataset path if needed
        config = BaseTransformerConfig()
        data_preparation = ModelPredictionPipeline(config)

        final_output_df = data_preparation.run_reconstruct_save_results_pipeline(
            code=request.code,
            forecast_horizon_list=request.forecast_horizon_list,
            lookback_list=request.lookback_list,
            data_path=request.data_path,
            #save_path=request.save_path
        )

        data_preparation.save_final_output_predictions(final_output_df, save_path=request.save_path)


        return JSONResponse(content={
            "status": "success",
            "message": "Model reconstruction pipeline triggered successfully!",
            "final_output_df": final_output_df.shape if final_output_df is not None else None
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering model reconstruction: {str(e)}")'''

@router.delete("/delete_old_data")
def delete_old_data():
    """
    Deletes old data from the production predictions dataset based on the logic:
    If the difference in days between target_date and forecast_date equals the forecast value, delete that row.
    
    Args:
        dataset_path (str): The path to the dataset from which old data should be deleted.
    
    Returns:
        str: The path to the updated dataset after deletion.
    """
    config = BaseTransformerConfig()
    pipeline = ModelPredictionPipeline(config)

    dataset_path = config.production_predictions_file
    metrics_df_path = config.production_metrics_file
    input_directory = config.data_path


    if not os.path.exists(dataset_path):
        print(f"No data file found at: {dataset_path}")
        return None

    try:    

        updated_path = pipeline.delete_old_data(predictions_dataset_path=dataset_path, real_data_dataset_path=input_directory, metrics_df_path=metrics_df_path)
        return JSONResponse(content={
            "status": "success",
            "message": f"Old data deleted successfully from: {updated_path}",
            "updated_dataset_path": updated_path
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting old data: {str(e)}")


@router.get("/read_data")
def read_data(data_path: str, code: str):
    """
    Reads the dataset and prepares it for plotting the time series trajectory and tendency analysis for a specific diagnostic code.
    
    Args:
        data_path (str): The path to the dataset file (e.g., CSV or Parquet).
        code (str): The diagnostic code for which to prepare the data.
    
    Returns:
        pd.DataFrame: The prepared dataset for the specified diagnostic code.
    """
    # Implementation for reading and preparing data
    pass
