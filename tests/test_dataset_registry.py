from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from fed_perso_xai.data.catalog import DatasetRegistry, DatasetSpec
from fed_perso_xai.orchestration.data_preparation import prepare_federated_dataset
from fed_perso_xai.utils.config import ArtifactPaths, DataPreparationConfig, PartitionConfig


def _build_paths(tmp_path):
    return ArtifactPaths(
        prepared_root=tmp_path / "prepared",
        partition_root=tmp_path / "datasets",
        centralized_root=tmp_path / "centralized",
        federated_root=tmp_path / "federated",
        comparison_root=tmp_path / "comparisons",
        cache_dir=tmp_path / "cache",
    )


def test_dataset_registry_extensibility(monkeypatch, tmp_path) -> None:
    def cleaning_hook(frame: pd.DataFrame) -> pd.DataFrame:
        cleaned = frame.copy()
        cleaned["city"] = cleaned["city"].replace({"?": "unknown"})
        return cleaned

    registry = DatasetRegistry()
    spec = DatasetSpec(
        key="toy_dataset",
        display_name="Toy Dataset",
        openml_data_id=999,
        target_transform=lambda value: int(value),
        target_column="target",
        cleaning_hook=cleaning_hook,
        feature_type_overrides={"binary_code": "categorical"},
        required_columns=("city", "income", "binary_code"),
    )
    registry.register(spec)

    frame = pd.DataFrame(
        {
            "city": ["A", "B", "?", "A", "B", "A", "B", "A", "B", "A", "B", "A"],
            "income": [10, 20, 15, 30, 18, 25, 12, 22, 16, 35, 14, 28],
            "binary_code": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    def fake_fetch_openml(*args, **kwargs):
        return SimpleNamespace(
            frame=frame,
            target=frame["target"].rename("target"),
            details={"name": "Toy Dataset", "version": "1"},
        )

    monkeypatch.setattr("fed_perso_xai.data.loaders.fetch_openml", fake_fetch_openml)

    result = prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="toy_dataset",
            seed=5,
            paths=_build_paths(tmp_path),
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        ),
        registry=registry,
    )

    assert result.prepared_artifacts.root_dir.exists()
    assert result.federated_artifacts.partition_metadata_path.exists()
    feature_metadata = result.prepared_artifacts.feature_metadata_path.read_text(encoding="utf-8")
    assert "binary_code" in feature_metadata
    assert registry.get("toy_dataset") == spec
    assert registry.list_keys() == ["toy_dataset"]

    with pytest.raises(ValueError):
        registry.register(spec)



def test_default_dataset_registry_includes_large_adult_dataset() -> None:
    from fed_perso_xai.data.catalog import DEFAULT_DATASET_REGISTRY

    spec = DEFAULT_DATASET_REGISTRY.get("adult_income_large")
    alias_spec = DEFAULT_DATASET_REGISTRY.get("cencus_income")

    assert spec.openml_data_id == 4535
    assert spec.target_column == "V42"
    assert spec.display_name == "Cencus Income"
    assert alias_spec == spec
    assert "224k" in spec.description


def test_default_dataset_registry_includes_loan_default_csv_dataset() -> None:
    from fed_perso_xai.data.catalog import DEFAULT_DATASET_REGISTRY

    spec = DEFAULT_DATASET_REGISTRY.get("loan_default")

    assert spec.source_type == "csv"
    assert spec.csv_path == "data/raw/loan_default/Loan_default.csv"
    assert spec.target_column == "Default"
    assert spec.row_id_column == "LoanID"
    assert spec.feature_type_overrides["LoanPurpose"] == "categorical"
    assert "credit-risk" in spec.description


def test_dataset_registry_supports_csv_backed_dataset(tmp_path) -> None:
    csv_path = tmp_path / "loan_default.csv"
    pd.DataFrame(
        {
            "LoanID": ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12"],
            "Income": [10, 20, 15, 30, 18, 25, 12, 22, 16, 35, 14, 28],
            "LoanPurpose": [
                "Business",
                "Home",
                "Auto",
                "Business",
                "Home",
                "Auto",
                "Business",
                "Home",
                "Auto",
                "Business",
                "Home",
                "Auto",
            ],
            "Default": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    ).to_csv(csv_path, index=False)

    registry = DatasetRegistry()
    spec = DatasetSpec(
        key="loan_default_csv",
        display_name="Loan Default CSV",
        target_transform=lambda value: int(value),
        source_type="csv",
        csv_path=str(csv_path),
        target_column="Default",
        row_id_column="LoanID",
        required_columns=("Income", "LoanPurpose"),
    )
    registry.register(spec)

    result = prepare_federated_dataset(
        DataPreparationConfig(
            dataset_name="loan_default_csv",
            seed=5,
            paths=_build_paths(tmp_path),
            partition=PartitionConfig(num_clients=3, alpha=1.0, min_client_samples=2, max_retries=20),
        ),
        registry=registry,
    )

    assert result.prepared_artifacts.root_dir.exists()
    assert result.federated_artifacts.partition_metadata_path.exists()
    assert registry.get("loan_default_csv") == spec
