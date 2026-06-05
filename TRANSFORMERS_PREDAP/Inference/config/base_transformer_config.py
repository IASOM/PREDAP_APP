"""Base Transformer configuration used by the inference bundle."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from sklearn.preprocessing import FunctionTransformer


@dataclass
class BaseTransformerConfig(ABC):
    lookback: int = 14
    forecast: int = 7
    code: str = "J00"

    head_size: int = 64
    num_heads: int = 8
    ff_dim: int = 512
    num_transformer_blocks: int = 4
    mlp_units: List[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.25
    activation_function: str = "gelu"

    learning_rate: float = 1e-4
    lr_max_multiplier: float = 100
    lr_min_multiplier: float = 10
    lr_warmup_ratio: float = 0.2
    epochs: int = 200
    batch_size: int = 256
    early_stop_patience: int = 50
    shuffle_data: bool = True
    save_train_history: bool = True

    data_path: str = "../data/FINAL_DB/full_CAT1.parquet"
    diagnostic_covariates_path: str = "../data/best_features/BEST_features_NOSMOOTH_"
    production_predictions_dir: str = "../production_predictions/final_output_predictions"
    production_predictions_file: str = "../production_predictions/final_output_predictions.parquet"
    production_metrics_file: str = "../production_predictions/production_evaluation_metrics.parquet"

    cutoff_date: str = "2008-01-01"
    final_cutoff_date: str = "2027-09-30"
    positional_encoding: bool = True
    default_split_ratio: float = 0.8
    eliminate_covid_data: bool = False
    covid_dates: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("2020-03-01", "2020-06-30"),
            ("2020-10-01", "2020-12-31"),
            ("2021-01-01", "2021-03-31"),
            ("2021-04-01", "2021-06-30"),
        ]
    )

    covid_token: bool = True
    evaluate_model: bool = False
    scaler: FunctionTransformer = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)

    plots_dir: str = "plots"
    model_folder: str = "../transformer_outputs/models_covid_token"

    CODES_LIST: List[str] = field(default_factory=lambda: ["J00", "T14", "M54"])
    LOOKBACK_LIST: List[int] = field(default_factory=lambda: [7, 14, 30, 182])
    FORECAST_LIST: List[int] = field(default_factory=lambda: [7, 14, 30, 182, 365])
    HEAD_SIZE_LIST: List[int] = field(default_factory=lambda: [2, 4, 8])
    NUM_HEADS_LIST: List[int] = field(default_factory=lambda: [4, 8])
    ACTIVATIONS_LIST: List[Any] = field(default_factory=lambda: ["gelu", "tanh", "leaky_relu"])
    COVID_TOKEN_LIST: List[bool] = field(default_factory=lambda: [False, True])
    FF_DIM_LIST: List[int] = field(default_factory=lambda: [32, 64])
    MLP_UNITS_LIST: List[int] = field(default_factory=lambda: [16, 32, 64, 128])

    DEFAULT_RESIDUAL_TRANSFORMER_PARAMS: dict = field(
        default_factory=lambda: {
            "head_size": 16,
            "num_heads": 16,
            "ff_dim": 512,
            "mlp_units": [256, 128],
            "num_transformer_blocks": 2,
            "dropout": 0.5,
        }
    )
    DEFAULT_RESIDUAL_LSTM_PARAMS: dict = field(
        default_factory=lambda: {
            "units_1": 128,
            "units_2": 64,
            "dropout": 0.2,
            "return_sequences": True,
        }
    )
    DEFAULT_RESIDUAL_SAVE_PARAMS: dict = field(
        default_factory=lambda: {"save_history": True, "save_model": True, "save_memory": False}
    )
    DEFAULT_RESIDUAL_TRAINING_PARAMS: dict = field(
        default_factory=lambda: {
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.1,
            "shuffle": False,
            "patience": 25,
        }
    )
    DEFAULT_SEASONAL_CATEGORICAL_VARS: list = field(
        default_factory=lambda: [
            "Day_of_Week",
            "Month",
            "Season",
            "Holiday",
            "School_Vacation",
            "Is_Weekend",
        ]
    )
    PANDEMIC_WAVES: dict = field(
        default_factory=lambda: {
            "Primera Onada": ("2020-03", "2020-06"),
            "Segona Onada": ("2020-10", "2020-12"),
            "Tercera Onada": ("2021-01", "2021-03"),
            "Quarta Onada": ("2021-04", "2021-06"),
        }
    )
    MEMORY_LOG_FILE: str = "memory.csv"

    def __post_init__(self):
        self._validate_core_parameters()
        self._validate_architecture_parameters()
        self._validate_training_parameters()

    def _validate_core_parameters(self):
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")
        if self.forecast <= 0:
            raise ValueError("forecast must be positive")
        if not self.code:
            raise ValueError("code cannot be empty")

    def _validate_architecture_parameters(self):
        if self.head_size <= 0:
            raise ValueError("head_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")

    def _validate_training_parameters(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lookback": self.lookback,
            "forecast": self.forecast,
            "code": self.code,
            "head_size": self.head_size,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "mlp_units": self.mlp_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "lr_max_multiplier": self.lr_max_multiplier,
            "lr_min_multiplier": self.lr_min_multiplier,
            "lr_warmup_ratio": self.lr_warmup_ratio,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "cutoff_date": self.cutoff_date,
            "covid_token": self.covid_token,
            "positional_encoding": self.positional_encoding,
            "activation_function": self.activation_function,
            "evaluate_model": self.evaluate_model,
            "data_path": self.data_path,
        }

    def print_config(self):
        class_name = self.__class__.__name__
        print(f"\n{'=' * 60}")
        print(f"{class_name.upper()}")
        print(f"{'=' * 60}")
        for key, value in self.to_dict().items():
            print(f"{key:25}: {value}")
        print(f"{'=' * 60}")

    def get_model_name(self) -> str:
        return (
            f"{self.code}_base_transformer_"
            f"{self.forecast}fh_{self.ff_dim}ff_{self.lookback}lb_"
            f"{self.learning_rate}lr.keras"
        )

    def get_diagnostic_residual_model_name(self) -> str:
        return (
            f"{self.code}_DIAGNOSTIC_RESIDUALS_LEARNING_"
            f"{self.forecast}fh_{self.ff_dim}ff_{self.lookback}lb_"
            f"{self.learning_rate}initlr.keras"
        )

    def get_seasonal_residual_model_name(self) -> str:
        return (
            f"{self.code}_SEASONAL_RESIDUALS_LEARNING_"
            f"{self.forecast}fh_{self.ff_dim}ff_{self.lookback}lb_"
            f"{self.learning_rate}initlr.keras"
        )
