"""Utilities for restoring teacher transformer models and distilling them into smaller students.

This module is designed to work with the grid-search configuration in
``conf/grid_search_V1.yaml``. It can:

1. Resolve one or more teacher configurations from the YAML file.
2. Restore trained teacher models from a local model folder or MLflow.
3. Build a smaller student transformer with the same input/output signature.
4. Distill the teacher into the student using blended soft targets.
5. Evaluate teacher and student with the same metrics used in the grid-search scripts.
6. Save the distilled model plus comparison artifacts in a structured folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from src.config.base_transformer_config import BaseTransformerConfig
from model_architechture.model_architecture_univ_transformer import (
	PositionalEncoding,
	RevIN,
	build_model,
)
from data_utils import data_preparation
from src.utils.experiments_utils import (
    smart_read
	)

pd.read_csv = smart_read

DEFAULT_CONFIG = BaseTransformerConfig()
CUSTOM_OBJECTS = {
	"PositionalEncoding": PositionalEncoding,
	"RevIN": RevIN,
}


@dataclass
class TeacherModelSpec:
	"""Resolved configuration used to restore a single teacher model."""

	code: str
	lookback: int
	forecast: int
	head_size: int
	num_heads: int
	ff_dim: int
	num_transformer_blocks: int
	mlp_units: List[int]
	dropout: float
	learning_rate: float
	activation_function: str
	covid_token: bool
	cutoff_date: str
	final_cutoff_date: str
	positional_encoding: bool
	data_path: str
	model_folder: str
	experiment_name: str
	tracking_uri: str
	batch_size: int
	evaluate_model: bool
	eliminate_covid_data: bool
	covid_dates: List[Sequence[str]]

	@property
	def teacher_model_name(self) -> str:
		return (
			f"{self.code}_base_transformer_"
			f"{self.forecast}fh_{self.ff_dim}ff_{self.lookback}lb_"
			f"{self.learning_rate}lr.keras"
		)

	@property
	def distilled_slug(self) -> str:
		return (
			f"{self.code}_lb{self.lookback}_fh{self.forecast}_"
			f"ff{self.ff_dim}_lr{self.learning_rate}"
		)


@dataclass
class DistillationMetrics:
	"""Metric bundle for one model and one split."""

	loss: float
	mae: float
	mse: float
	rmse: float
	wape: float


@dataclass
class DistillationRunResult:
	"""Summary of a completed teacher/student distillation run."""

	teacher_name: str
	student_name: str
	teacher_params: int
	student_params: int
	compression_ratio: float
	teacher_metrics: Dict[str, float]
	student_metrics: Dict[str, float]
	output_dir: str
	model_path: str
	teacher_model_source: str


def _parse_scalar_or_list(value: Any) -> List[str]:
	"""Parse a Hydra-style scalar or CSV-like list into a normalized string list."""

	if value is None:
		return []

	if isinstance(value, (list, tuple)):
		return [str(item).strip().strip('"').strip("'") for item in value if str(item).strip()]

	if isinstance(value, str):
		text = value.strip()
		if not text:
			return []
		if "," in text:
			reader = csv.reader([text], skipinitialspace=True)
			return [item.strip().strip('"').strip("'") for item in next(reader) if item.strip()]
		return [text.strip('"').strip("'")]

	return [str(value)]


def _as_bool(value: Any) -> bool:
	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return bool(value)
	return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _ensure_directory(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
	with config_path.open("r", encoding="utf-8") as handle:
		return yaml.safe_load(handle)


def _experiment_pairs(config: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
	pairs = config.get("experiment_pairs", {}) or {}
	normalized: Dict[str, Dict[str, int]] = {}
	for setup_name, values in pairs.items():
		normalized[str(setup_name)] = {"lb": int(values["lb"]), "fc": int(values["fc"])}
	return normalized


def _resolve_teacher_specs(config: Dict[str, Any], distill_all_configs: bool = False) -> List[TeacherModelSpec]:
	"""Create one or more teacher specs from the YAML configuration."""

	model_cfg = config.get("model", {}) or {}
	training_cfg = config.get("training", {}) or {}
	data_cfg = config.get("data", {}) or {}
	mlflow_cfg = config.get("mlflow", {}) or {}
	hydra_cfg = config.get("hydra", {}) or {}
	sweeper_cfg = hydra_cfg.get("sweeper", {}) or {}
	sweep_params = sweeper_cfg.get("params", {}) or {}

	experiment_pairs = _experiment_pairs(config)
	default_setup = config.get("experiment_setup", next(iter(experiment_pairs.keys()), None))

	if distill_all_configs:
		setup_values = _parse_scalar_or_list(config.get("experiment_setup", default_setup))
		if not setup_values and default_setup is not None:
			setup_values = [str(default_setup)]

		target_codes = _parse_scalar_or_list(sweep_params.get("model.target_code", model_cfg.get("target_code")))
		if not target_codes:
			target_codes = [str(model_cfg.get("target_code", "B34"))]
	else:
		setup_values = [str(default_setup)] if default_setup is not None else []
		target_codes = [str(model_cfg.get("target_code", "B34"))]

	specs: List[TeacherModelSpec] = []
	for code, setup in product(target_codes, setup_values):
		if setup not in experiment_pairs:
			continue

		pair = experiment_pairs[setup]
		specs.append(
			TeacherModelSpec(
				code=code,
				lookback=int(pair["lb"]),
				forecast=int(pair["fc"]),
				head_size=int(model_cfg.get("head_size", 32)),
				num_heads=int(model_cfg.get("num_heads", 8)),
				ff_dim=int(model_cfg.get("ff_dim", 512)),
				num_transformer_blocks=int(model_cfg.get("num_transformer_blocks", 2)),
				mlp_units=[int(item) for item in model_cfg.get("mlp_units", [512, 256])],
				dropout=float(model_cfg.get("dropout", 0.5)),
				learning_rate=float(model_cfg.get("learning_rate", 1e-5)),
				activation_function=str(model_cfg.get("activation", model_cfg.get("activation_function", "gelu"))),
				covid_token=_as_bool(model_cfg.get("covid_token", True)),
				cutoff_date=str(training_cfg.get("cutoff_date", "2008-01-01")),
				final_cutoff_date=str(training_cfg.get("final_cutoff_date", DEFAULT_CONFIG.final_cutoff_date)),
				positional_encoding=_as_bool(training_cfg.get("positional_encoding", True)),
				data_path=str(data_cfg.get("data_path", DEFAULT_CONFIG.data_path)),
				model_folder=str(model_cfg.get("model_folder", DEFAULT_CONFIG.model_folder)),
				experiment_name=str(mlflow_cfg.get("experiment_name", "predap_distillation")),
				tracking_uri=str(mlflow_cfg.get("tracking_uri", "file:./mlruns")),
				batch_size=int(model_cfg.get("batch_size", DEFAULT_CONFIG.batch_size)),
				evaluate_model=_as_bool(training_cfg.get("evaluate_model", True)),
				eliminate_covid_data=_as_bool(training_cfg.get("eliminate_covid_data", False)),
				covid_dates=list(training_cfg.get("covid_dates", DEFAULT_CONFIG.covid_dates)),
			)
		)

	return specs


def _load_teacher_from_local(model_folder: Path, model_name: str) -> Optional[tf.keras.Model]:
	candidate = model_folder / model_name
	if candidate.exists():
		return tf.keras.models.load_model(candidate, custom_objects=CUSTOM_OBJECTS, compile=False)
	return None


def _load_teacher_from_mlflow(spec: TeacherModelSpec, model_artifact_path: str = "univariate_transformer") -> tf.keras.Model:
	mlflow.set_tracking_uri(spec.tracking_uri)
	try:
		mlflow.set_experiment(spec.experiment_name)
	except Exception:
		pass

	filter_parts = [
		f"params.target_code = '{spec.code}'",
		f"params.lookback = '{spec.lookback}'",
		f"params.forecast_horizon = '{spec.forecast}'",
		f"params.activation_function = '{spec.activation_function}'",
		f"params.covid_token = '{str(spec.covid_token)}'",
	]
	filter_string = " and ".join(filter_parts)

	runs = mlflow.search_runs(
		experiment_names=[spec.experiment_name],
		filter_string=filter_string,
		order_by=["start_time DESC"],
		max_results=1,
	)

	if runs.empty:
		raise FileNotFoundError(
			f"No MLflow run found for {spec.code} lb={spec.lookback} fh={spec.forecast}"
		)

	run_id = runs.iloc[0]["run_id"]
	local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=model_artifact_path)
	search_root = Path(local_path)

	candidate_paths = list(search_root.rglob("*.keras"))
	if not candidate_paths:
		raise FileNotFoundError(f"No .keras file found under MLflow artifact path: {search_root}")

	preferred = [path for path in candidate_paths if path.name == "model.keras"]
	model_path = preferred[0] if preferred else candidate_paths[0]
	return tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)


def _restore_teacher_model(spec: TeacherModelSpec, model_artifact_path: str = "univariate_transformer") -> tuple[tf.keras.Model, str]:
	local_folder = Path(spec.model_folder)
	_ensure_directory(local_folder)

	local_model = _load_teacher_from_local(local_folder, spec.teacher_model_name)
	if local_model is not None:
		return local_model, str(local_folder / spec.teacher_model_name)

	teacher = _load_teacher_from_mlflow(spec, model_artifact_path=model_artifact_path)
	return teacher, f"mlflow://{spec.experiment_name}/{spec.teacher_model_name}"


def _smaller_student_spec(spec: TeacherModelSpec, compression_factor: float = 0.5) -> Dict[str, Any]:
	"""Create a reduced architecture that remains compatible with the teacher inputs."""

	head_size = max(4, int(spec.head_size * compression_factor))
	num_heads = max(1, int(spec.num_heads * compression_factor))
	ff_dim = max(16, int(spec.ff_dim * (compression_factor ** 2)))
	num_transformer_blocks = max(1, int(round(spec.num_transformer_blocks * compression_factor)))

	mlp_units: List[int] = []
	for unit in spec.mlp_units[:2]:
		compressed = max(8, int(unit * (compression_factor ** 2)))
		if compressed not in mlp_units:
			mlp_units.append(compressed)

	if not mlp_units:
		mlp_units = [max(8, int(max(spec.mlp_units) * (compression_factor ** 2)))]

	return {
		"head_size": head_size,
		"num_heads": num_heads,
		"ff_dim": ff_dim,
		"num_transformer_blocks": num_transformer_blocks,
		"mlp_units": mlp_units,
	}


def _build_student_model(spec: TeacherModelSpec, input_shape: Sequence[int], compression_factor: float = 0.5) -> tf.keras.Model:
	architecture = _smaller_student_spec(spec, compression_factor=compression_factor)
	return build_model(
		input_shape,
		head_size=architecture["head_size"],
		num_heads=architecture["num_heads"],
		ff_dim=architecture["ff_dim"],
		num_transformer_blocks=architecture["num_transformer_blocks"],
		mlp_units=architecture["mlp_units"],
		activation_function=spec.activation_function,
		dropout=spec.dropout,
		mlp_dropout=spec.dropout,
		n_pred=spec.forecast,
		pos_encoding=spec.positional_encoding,
	)


def _prepare_data_for_spec(spec: TeacherModelSpec) -> Dict[str, np.ndarray]:
	train_x, train_y = data_preparation.prepare_data(
		spec.data_path,
		spec.code,
		spec.lookback,
		spec.forecast,
		cutoff_date=spec.cutoff_date,
		max_date=spec.final_cutoff_date,
		covid_token=spec.covid_token,
		train=True,
		debug=False,
		univariate=True,
		scaler=DEFAULT_CONFIG.scaler,
		eliminate_covid_data=spec.eliminate_covid_data,
		covid_dates=spec.covid_dates,
	)

	test_x, test_y = data_preparation.prepare_data(
		spec.data_path,
		spec.code,
		spec.lookback,
		spec.forecast,
		cutoff_date=spec.cutoff_date,
		max_date=spec.final_cutoff_date,
		covid_token=spec.covid_token,
		train=False,
		debug=False,
		univariate=True,
		scaler=DEFAULT_CONFIG.scaler,
		eliminate_covid_data=spec.eliminate_covid_data,
		covid_dates=spec.covid_dates,
	)

	train_x_orig, train_y_orig = data_preparation.prepare_data_not_normalized(
		spec.data_path,
		spec.code,
		spec.lookback,
		spec.forecast,
		cutoff_date=spec.cutoff_date,
		max_date=spec.final_cutoff_date,
		covid_token=spec.covid_token,
		train=True,
		debug=False,
		univariate=True,
		eliminate_covid_data=spec.eliminate_covid_data,
		covid_dates=spec.covid_dates,
	)

	test_x_orig, test_y_orig = data_preparation.prepare_data_not_normalized(
		spec.data_path,
		spec.code,
		spec.lookback,
		spec.forecast,
		cutoff_date=spec.cutoff_date,
		max_date=spec.final_cutoff_date,
		covid_token=spec.covid_token,
		train=False,
		debug=False,
		univariate=True,
		eliminate_covid_data=spec.eliminate_covid_data,
		covid_dates=spec.covid_dates,
	)

	return {
		"train_x": train_x,
		"train_y": train_y,
		"test_x": test_x,
		"test_y": test_y,
		"train_x_orig": train_x_orig,
		"train_y_orig": train_y_orig,
		"test_x_orig": test_x_orig,
		"test_y_orig": test_y_orig,
	}


def _compile_for_evaluation(model: tf.keras.Model) -> tf.keras.Model:
	model.compile(loss="mae", metrics=["mae", "mse"], optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
	return model


def _compute_metrics(
	spec: TeacherModelSpec,
	model: tf.keras.Model,
	test_x: np.ndarray,
	test_y_normalized: np.ndarray,
	test_y_orig: np.ndarray,
) -> DistillationMetrics:
	"""Compute the same comparison metrics used in the training scripts."""

	loss, _, _ = model.evaluate(test_x, test_y_normalized, verbose=0)
	predictions = model.predict(test_x, verbose=0)
	clipped_predictions = np.maximum(predictions, 0)

	original_scale_df = pd.read_parquet(spec.data_path)
	predictions_original_scale = data_preparation.inverse_transform_predictions(
		clipped_predictions,
		original_scale_df,
		code=spec.code,
		forecast=spec.forecast,
		lookback=spec.lookback,
		cutoff_date=spec.cutoff_date,
		max_date=spec.final_cutoff_date,
		scaler=DEFAULT_CONFIG.scaler,
		eliminate_covid_data=spec.eliminate_covid_data,
		covid_dates=spec.covid_dates,
	)

	mae = float(mean_absolute_error(test_y_orig, predictions_original_scale))
	mse = float(mean_squared_error(test_y_orig, predictions_original_scale))
	rmse = float(np.sqrt(mse))
	denominator = float(np.sum(np.abs(test_y_orig))) + 1e-8
	wape = float(np.sum(np.abs(test_y_orig - predictions_original_scale)) / denominator * 100.0)

	return DistillationMetrics(
		loss=float(loss),
		mae=mae,
		mse=mse,
		rmse=rmse,
		wape=wape,
	)


def _evaluate_model_pair(
	spec: TeacherModelSpec,
	teacher_model: tf.keras.Model,
	student_model: tf.keras.Model,
	dataset: Dict[str, np.ndarray],
) -> tuple[DistillationMetrics, DistillationMetrics]:
	teacher_model = _compile_for_evaluation(teacher_model)
	student_model = _compile_for_evaluation(student_model)

	teacher_metrics = _compute_metrics(
		spec,
		teacher_model,
		dataset["test_x"],
		dataset["test_y"],
		dataset["test_y_orig"],
	)
	student_metrics = _compute_metrics(
		spec,
		student_model,
		dataset["test_x"],
		dataset["test_y"],
		dataset["test_y_orig"],
	)
	return teacher_metrics, student_metrics


def _fit_student_model(
	student_model: tf.keras.Model,
	train_x: np.ndarray,
	train_y: np.ndarray,
	teacher_predictions: np.ndarray,
	learning_rate: float,
	epochs: int = 3000,
	batch_size: int = 32,
	distillation_alpha: float = 0.3,
) -> tf.keras.callbacks.History:
	"""Train the student on blended labels from the teacher and the true targets."""

	blended_targets = (1.0 - distillation_alpha) * train_y + distillation_alpha * teacher_predictions
	student_model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss="mae",
		metrics=["mae", "mse"],
	)

	early_stop = tf.keras.callbacks.EarlyStopping(
		monitor="val_loss",
		patience=30,
		restore_best_weights=True,
		verbose=1,
	)

	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
		monitor="val_loss",
		factor=0.5,
		patience=5,
		min_lr=1e-7,
		verbose=1,
	)

	history = student_model.fit(
		train_x,
		blended_targets,
		validation_split=0.1,
		shuffle=False,
		epochs=epochs,
		batch_size=batch_size,
		callbacks=[early_stop, reduce_lr],
		verbose=1,
	)
	return history


def _save_distilled_artifacts(
	output_root: Path,
	spec: TeacherModelSpec,
	student_model: tf.keras.Model,
	teacher_metrics: DistillationMetrics,
	student_metrics: DistillationMetrics,
	teacher_model_name: str,
	teacher_model_source: str,
	teacher_params: int,
	student_params: int,
) -> DistillationRunResult:
	run_dir = output_root / spec.distilled_slug
	_ensure_directory(run_dir)

	model_path = run_dir / "distilled_model.keras"
	student_model.save(model_path)

	metrics_payload = {
		"teacher": asdict(teacher_metrics),
		"student": asdict(student_metrics),
		"teacher_model_name": teacher_model_name,
		"teacher_model_source": teacher_model_source,
		"teacher_params": teacher_params,
		"student_params": student_params,
		"compression_ratio": float(student_params / max(teacher_params, 1)),
		"spec": asdict(spec),
	}

	with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
		json.dump(metrics_payload, handle, indent=2)

	with (run_dir / "spec.json").open("w", encoding="utf-8") as handle:
		json.dump(asdict(spec), handle, indent=2)

	pd.DataFrame(
		[
			{"model": "teacher", **asdict(teacher_metrics)},
			{"model": "student", **asdict(student_metrics)},
		]
	).to_csv(run_dir / "comparison.csv", index=False)

	return DistillationRunResult(
		teacher_name=teacher_model_name,
		student_name=model_path.name,
		teacher_params=teacher_params,
		student_params=student_params,
		compression_ratio=float(student_params / max(teacher_params, 1)),
		teacher_metrics=asdict(teacher_metrics),
		student_metrics=asdict(student_metrics),
		output_dir=str(run_dir),
		model_path=str(model_path),
		teacher_model_source=teacher_model_source,
	)


class ModelDistillationPipeline:
	"""End-to-end distillation pipeline for teacher/student transformer models."""

	def __init__(
		self,
		config_path: str | Path = REPO_ROOT / "conf" / "grid_search_V1.yaml",
		output_dir: str | Path = REPO_ROOT / "distilled_models",
		distillation_alpha: float = 0.5,
		compression_factor: float = 0.7,
		model_artifact_path: str = "univariate_transformer",
		distill_all_configs: bool = False,
		epochs: int = 3000,
		batch_size: Optional[int] = None,
	) -> None:
		self.config_path = Path(config_path)
		self.output_dir = Path(output_dir)
		self.distillation_alpha = distillation_alpha
		self.compression_factor = compression_factor
		self.model_artifact_path = model_artifact_path
		self.distill_all_configs = distill_all_configs
		self.epochs = epochs
		self.batch_size_override = batch_size

		self.config = _load_yaml_config(self.config_path)
		self.specs = _resolve_teacher_specs(self.config, distill_all_configs=distill_all_configs)

		_ensure_directory(self.output_dir)

	def run(self) -> List[DistillationRunResult]:
		results: List[DistillationRunResult] = []

		for spec in self.specs:
			print("\n" + "=" * 80)
			print(f"Distilling teacher model for {spec.code} | lb={spec.lookback} | fh={spec.forecast}")
			print("=" * 80)

			dataset = _prepare_data_for_spec(spec)

			teacher_model, teacher_source = _restore_teacher_model(spec, model_artifact_path=self.model_artifact_path)
			teacher_model = _compile_for_evaluation(teacher_model)
			teacher_predictions_train = teacher_model.predict(dataset["train_x"], verbose=0)

			student_model = _build_student_model(
				spec,
				input_shape=dataset["train_x"].shape[1:],
				compression_factor=self.compression_factor,
			)

			fit_batch_size = self.batch_size_override or spec.batch_size
			_fit_student_model(
				student_model=student_model,
				train_x=dataset["train_x"],
				train_y=dataset["train_y"],
				teacher_predictions=teacher_predictions_train,
				learning_rate=spec.learning_rate,
				epochs=self.epochs,
				batch_size=fit_batch_size,
				distillation_alpha=self.distillation_alpha,
			)

			teacher_metrics, student_metrics = _evaluate_model_pair(spec, teacher_model, student_model, dataset)

			teacher_params = teacher_model.count_params()
			student_params = student_model.count_params()

			result = _save_distilled_artifacts(
				output_root=self.output_dir,
				spec=spec,
				student_model=student_model,
				teacher_metrics=teacher_metrics,
				student_metrics=student_metrics,
				teacher_model_name=spec.teacher_model_name,
				teacher_model_source=teacher_source,
				teacher_params=teacher_params,
				student_params=student_params,
			)
			results.append(result)

			print(f"Teacher metrics: {result.teacher_metrics}")
			print(f"Student metrics: {result.student_metrics}")
			print(f"Saved distilled model to: {result.model_path}")

			tf.keras.backend.clear_session()

		self._save_summary(results)
		return results

	def _save_summary(self, results: List[DistillationRunResult]) -> None:
		summary_rows = []
		for result in results:
			summary_rows.append(
				{
					"teacher_name": result.teacher_name,
					"student_name": result.student_name,
					"teacher_params": result.teacher_params,
					"student_params": result.student_params,
					"compression_ratio": result.compression_ratio,
					"teacher_mae": result.teacher_metrics["mae"],
					"teacher_mse": result.teacher_metrics["mse"],
					"teacher_rmse": result.teacher_metrics["rmse"],
					"teacher_wape": result.teacher_metrics["wape"],
					"student_mae": result.student_metrics["mae"],
					"student_mse": result.student_metrics["mse"],
					"student_rmse": result.student_metrics["rmse"],
					"student_wape": result.student_metrics["wape"],
					"output_dir": result.output_dir,
				}
			)

		summary_df = pd.DataFrame(summary_rows)
		summary_df.to_csv(self.output_dir / "distillation_summary.csv", index=False)
		summary_df.to_json(self.output_dir / "distillation_summary.json", orient="records", indent=2)


def build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Restore transformer teachers and distill them into smaller students.")
	parser.add_argument(
		"--config-path",
		type=str,
		default=str(REPO_ROOT / "conf" / "grid_search_V1.yaml"),
		help="Path to the grid-search YAML configuration.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(REPO_ROOT / "distilled_models"),
		help="Directory where distilled models and summaries will be saved.",
	)
	parser.add_argument(
		"--distillation-alpha",
		type=float,
		default=0.3,
		help="Weight of the teacher predictions in the blended training targets.",
	)
	parser.add_argument(
		"--compression-factor",
		type=float,
		default=0.5,
		help="Compression factor used to shrink the teacher architecture.",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=1000,
		help="Maximum number of training epochs for the student.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=None,
		help="Optional batch size override for student training.",
	)
	parser.add_argument(
		"--model-artifact-path",
		type=str,
		default="univariate_transformer",
		help="MLflow artifact path that contains the teacher model.",
	)
	parser.add_argument(
		"--distill-all-configs",
		action="store_true",
		help="Expand the YAML sweep lists and distill every resolved teacher configuration.",
	)
	return parser


def main() -> List[DistillationRunResult]:
	parser = build_argument_parser()
	args = parser.parse_args()

	pipeline = ModelDistillationPipeline(
		config_path=args.config_path,
		output_dir=args.output_dir,
		distillation_alpha=args.distillation_alpha,
		compression_factor=args.compression_factor,
		model_artifact_path=args.model_artifact_path,
		distill_all_configs=args.distill_all_configs,
		epochs=args.epochs,
		batch_size=args.batch_size,
	)
	return pipeline.run()


if __name__ == "__main__":
	main()
