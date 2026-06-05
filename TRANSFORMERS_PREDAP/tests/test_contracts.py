from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
