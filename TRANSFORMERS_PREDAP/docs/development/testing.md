# Testing

Predap uses **pytest** for unit testing and integration testing.

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_data_preparation.py

# Run specific test function
pytest tests/test_data_preparation.py::test_rolling_sequences -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Test Categories

### Unit Tests

Test individual functions in isolation:

```python
# tests/test_data_preparation.py
import numpy as np
from src.utils.data_preparation import (
    split_train_test,
    add_covid_token,
    eliminate_covid_dates,
)

def test_split_train_test():
    """Train/test split respects the specified ratio."""
    import pandas as pd
    df = pd.DataFrame({"a": range(100)})
    train, test = split_train_test(df, split_ratio=0.8)
    assert len(train) == 80
    assert len(test) == 20

def test_add_covid_token():
    """COVID token column is correctly added."""
    import pandas as pd
    dates = pd.date_range("2019-01-01", "2022-12-31", freq="D")
    df = pd.DataFrame({"timestamp": dates, "value": range(len(dates))})
    result = add_covid_token(df)
    assert "covid_token" in result.columns
    assert result["covid_token"].sum() > 0  # Some dates should be flagged
```

### Model Architecture Tests

Verify model builds correctly with expected shapes:

```python
# tests/test_model_architecture.py
import tensorflow as tf
from src.univariate_transformer.model_architecture_univ_transformer import build_model

def test_build_model_output_shape():
    """Model output shape matches forecast horizon."""
    model = build_model(
        input_shape=(14, 10),
        head_size=32,
        num_heads=4,
        ff_dim=256,
        num_transformer_blocks=2,
        mlp_units=[256],
        activation_function="gelu",
        dropout=0.1,
        mlp_dropout=0.1,
        n_pred=7,
        pos_encoding=True
    )
    assert model.output_shape == (None, 7)

def test_model_forward_pass():
    """Model can process a batch without errors."""
    model = build_model(
        input_shape=(14, 10),
        head_size=32, num_heads=4, ff_dim=256,
        num_transformer_blocks=2, mlp_units=[256],
        activation_function="gelu", dropout=0.1,
        mlp_dropout=0.1, n_pred=7, pos_encoding=True
    )
    X = tf.random.normal((8, 14, 10))
    output = model(X, training=False)
    assert output.shape == (8, 7)
```

### Configuration Tests

```python
# tests/test_config.py
from src.config.base_transformer_config import BaseTransformerConfig

def test_default_config():
    """Default config initializes without errors."""
    config = BaseTransformerConfig()
    assert config.lookback > 0
    assert config.forecast > 0

def test_model_name_generation():
    """Model names follow expected pattern."""
    config = BaseTransformerConfig()
    name = config.get_model_name()
    assert name.endswith(".keras")
    assert "base_transformer" in name
```

### Integration Tests

Test the full pipeline end-to-end (requires data):

```python
# tests/test_pipeline_integration.py
import pytest

@pytest.mark.integration
def test_univariate_pipeline():
    """Full univariate pipeline completes without errors."""
    from src.main_train_univ_transformer_class import (
        TransformerUnivConfig,
        UnivariateTransformerPipeline,
    )
    
    config = TransformerUnivConfig(
        lookback=7, forecast=3, code="J00",
        activation_function="gelu", covid_token=True,
        cutoff_date="2020-01-01",  # Short date range for speed
        head_size=16, num_heads=2, ff_dim=64,
        mlp_units=[64], learning_rate=1e-3, batch_size=32,
        data_path="data/test_data.parquet",
        epochs=1,  # Single epoch for speed
    )
    
    pipeline = UnivariateTransformerPipeline(config)
    outputs = pipeline.run_complete_pipeline()
    
    assert outputs.model is not None
    assert outputs.mse >= 0
```

---

## Test Fixtures

Common fixtures for test data generation:

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_timeseries():
    """Generate a sample time series DataFrame."""
    dates = pd.date_range("2010-01-01", "2024-12-31", freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "J00": np.random.poisson(50, len(dates)).astype(float),
        "B34": np.random.poisson(20, len(dates)).astype(float),
    })

@pytest.fixture
def sample_sequences():
    """Generate sample input/output sequences."""
    X = np.random.randn(100, 14, 10).astype(np.float32)
    Y = np.random.randn(100, 7).astype(np.float32)
    return X, Y
```

---

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest tests/ -v --cov=src
```
