from pydantic import BaseModel, Field
from typing import Optional, List


class AddNewDataRequest(BaseModel):
    new_data_path: Optional[str] = Field(
        default=None,
        description="Path to the .parquet file containing the source data. Uses config default if not provided."
    )
    cutoff_date: Optional[str] = Field(
        default=None,
        description="The starting boundary for data processing (e.g., '2008-01-01'). Uses config default if not provided."
    )
    max_date: Optional[str] = Field(
        default=None,
        description="The upper boundary for the dataset timeline (e.g., '2025-09-30'). Uses config default if not provided."
    )
    eliminate_covid_data: Optional[bool] = Field(
        default=None,
        description="If True, excludes 2020-2022 from mean calculations. Uses config default if not provided."
    )
    covid_token: Optional[bool] = Field(
        default=None,
        description="If True, adds/maintains a boolean flag for COVID periods. Uses config default if not provided."
    )
    provided_data: Optional[List[float]] = Field(
        default=None,
        description="An array of values to use for the new row. Must match the number of feature columns."
    )
    save_path: Optional[str] = Field(
        default="../data/FINAL_DB1",
        description="Directory path where the updated file will be saved."
    )
    delete_old: Optional[bool] = Field(
        default=True,
        description="If True, deletes the existing file with the same name before saving."
    )


class ModelReconstructionRequest(BaseModel):
    code: str = Field(..., description="The code for which to reconstruct the model (e.g., 'code1').")
    lookback_list: List[int] = Field(..., description="List of lookback windows.")
    forecast_horizon_list: List[int] = Field(..., description="List of forecast horizons.")
    head_size: int = Field(..., description="The size of each attention head.")
    num_heads: int = Field(..., description="The number of attention heads.")
    ff_dim: int = Field(..., description="The dimensionality of the feedforward network.")
    num_transformer_blocks: int = Field(..., description="The number of transformer blocks.")
    mlp_units: List[int] = Field(..., description="The number of units in each MLP layer.")
    activation_function: str = Field(..., description="The activation function to use in the MLP layers.")
    dropout: float = Field(default=0.0, description="The dropout rate for regularization.")
    learning_rate: float = Field(default=0.001, description="The learning rate for training.")
    epochs: int = Field(default=50, description="The number of epochs to train the model.")
    batch_size: int = Field(default=32, description="The batch size for training.")
    cutoff_date: str = Field(..., description="The cutoff date for training data (e.g., '2025-09-30').")
    covid_token: bool = Field(default=True, description="If True, adds/maintains a boolean flag for COVID periods.")
    positional_encoding: bool = Field(default=True, description="If True, uses positional encoding in the model.")
    evaluate_model: bool = Field(default=True, description="If True, evaluates the model after training.")
    data_path: str = Field(..., description="The path to the input data directory.")
    old_data_path: Optional[str] = Field(
        default=None,
        description="Optional legacy-format data path used for diagnostic covariates. Defaults to data_path."
    )
    model_folder: Optional[str] = Field(
        default=None,
        description="Directory containing quantized model weights. Uses config default if omitted."
    )
    prediction_dates: Optional[List[str]] = Field(
        default=None,
        description="Specific prediction dates to run. Uses production defaults if omitted."
    )
    save_path: str = Field(..., description="The path to the directory where results will be saved.")
