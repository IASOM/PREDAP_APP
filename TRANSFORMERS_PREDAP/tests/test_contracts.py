from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def _function_node(source: str, function_name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"Function not found: {function_name}")


class ContractTests(unittest.TestCase):
    def test_lr_schedule_uses_dataclass_field_names(self) -> None:
        source = _read("src/config/base_transformer_config.py")
        function = _function_node(source, "get_lr_schedule_params")
        uppercase_attrs = {
            node.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Attribute) and node.attr.isupper()
        }

        self.assertEqual(uppercase_attrs, set())
        self.assertIn("self.learning_rate", source)
        self.assertIn("self.lr_max_multiplier", source)
        self.assertIn("self.epochs", source)

    def test_api_reconstruction_job_passes_required_pipeline_inputs(self) -> None:
        source = _read("api/routers/production.py")

        self.assertIn("config.max_date", source)
        self.assertNotIn("config.final_cutoff_date", source)
        self.assertIn("input_directory=input_directory", source)
        self.assertIn("old_input_directory=old_input_directory", source)
        self.assertIn("final_output_df=pd.DataFrame()", source)
        self.assertIn("prediction_dates=payload.get(\"prediction_dates\")", source)

    def test_reconstruction_schema_matches_config_shape(self) -> None:
        source = _read("api/schemas/production_schemas.py")

        self.assertIn("mlp_units: List[int]", source)
        self.assertIn("old_data_path: Optional[str]", source)
        self.assertIn("model_folder: Optional[str]", source)
        self.assertIn("prediction_dates: Optional[List[str]]", source)

    def test_save_predictions_respects_explicit_output_path(self) -> None:
        source = _read("production/model_reconstruction_pipeline.py")
        function = _function_node(source, "save_final_output_predictions")
        function_source = ast.get_source_segment(source, function) or ""

        self.assertIn(
            "output_path = output_path or self.config.production_predictions_dir",
            function_source,
        )
        self.assertNotIn("output_path = self.config.production_predictions_dir", function_source)
        self.assertIn("final_output_df cannot be None", function_source)

    def test_residual_pipelines_require_complete_prediction_handoff(self) -> None:
        for relative_path in (
            "src/main_train_diagnostic_residual_transformer_class.py",
            "src/main_train_seasonal_residual_transformer_class.py",
        ):
            source = _read(relative_path)
            function = _function_node(source, "prepare_base_model_data")
            function_source = ast.get_source_segment(source, function) or ""

            self.assertIn("self.config.predictions_train_corrected is None", function_source)
            self.assertIn("self.config.predictions_test_corrected is None", function_source)
            self.assertIn("load_base_model_transformer", function_source)

    def test_temporal_pairs_support_scalar_broadcasting(self) -> None:
        from predap_cli import _temporal_pairs

        self.assertEqual(
            _temporal_pairs(lookback=7, forecast=7, lookbacks=[7, 14], forecasts=None),
            [(7, 7), (14, 7)],
        )
        self.assertEqual(
            _temporal_pairs(lookback=7, forecast=7, lookbacks=None, forecasts=[7, 14]),
            [(7, 7), (7, 14)],
        )
        self.assertEqual(
            _temporal_pairs(lookback=7, forecast=7, lookbacks=[7, 14], forecasts=[7, 14]),
            [(7, 7), (14, 14)],
        )

        with self.assertRaises(ValueError):
            _temporal_pairs(lookback=7, forecast=7, lookbacks=[7, 14], forecasts=[7, 14, 30])

    def test_prediction_output_keeps_temporal_parameter_identity(self) -> None:
        source = _read("production/model_reconstruction_pipeline.py")
        function = _function_node(source, "run_reconstruct_save_results_pipeline")
        function_source = ast.get_source_segment(source, function) or ""

        self.assertIn('auxiliary_output_df["lookback"] = lookback', function_source)
        self.assertIn('auxiliary_output_df["forecast"] = forecast', function_source)
        self.assertIn('auxiliary_output_df["forecast_step"] = np.arange(1, forecast + 1)', function_source)

    def test_prediction_writes_replace_matching_partitions(self) -> None:
        source = _read("production/model_reconstruction_pipeline.py")

        self.assertIn('existing_data_behavior="delete_matching"', source)
        self.assertNotIn('existing_data_behavior="overwrite_or_ignore"', source)

    def test_prediction_metrics_iterate_all_forecasts(self) -> None:
        source = _read("production/model_reconstruction_pipeline.py")
        function = _function_node(source, "compute_evaluation_metrics")
        function_source = ast.get_source_segment(source, function) or ""

        self.assertIn("np.unique(predictions_df['forecast'].values)", function_source)
        self.assertNotIn("np.unique(predictions_df['forecast'].values[0])", function_source)


if __name__ == "__main__":
    unittest.main()
