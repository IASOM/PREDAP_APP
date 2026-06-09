import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "VISI_SITUACIO_VISITA_N": "VISI_SITUACIO_VISITA_NO_PROGRAMADA",
    "VISI_SITUACIO_VISITA_P": "VISI_SITUACIO_VISITA_PROGRAMADA",
    "VISI_SITUACIO_VISITA_R": "VISI_SITUACIO_VISITA_URGENT",
    "SERVEI_CODI_MF": "SERVEI_CODI_MEDFAM",
}


def canonical_column_name(name: str) -> str:
    canonical = str(name).strip().replace("#", ":")
    if canonical.startswith("DEMAND_"):
        canonical = canonical[len("DEMAND_") :]
    canonical = canonical.replace("__", "_").upper()
    if canonical == "TOTAL":
        canonical = "DEMANDA_TOTAL"
    for legacy, current in COLUMN_ALIASES.items():
        canonical = re.sub(
            rf"(^|_){re.escape(legacy)}($|_)",
            lambda match: f"{match.group(1)}{current}{match.group(2)}",
            canonical,
        )
    canonical = re.sub(r"[^A-Z0-9]+", "_", canonical).strip("_")
    return canonical


def column_lookup(columns: Iterable[str]) -> dict:
    lookup = {}
    for column in columns:
        lookup.setdefault(canonical_column_name(column), column)
    return lookup


def resolve_column(columns: Iterable[str], requested: str, role: str = "column") -> str:
    columns = list(columns)
    if requested in columns:
        return requested
    lookup = column_lookup(columns)
    resolved = lookup.get(canonical_column_name(requested))
    if resolved is not None:
        if resolved != requested:
            print(f"-> INFO: Mapped {role} '{requested}' to dataset column '{resolved}'.")
        return resolved
    sample = ", ".join(map(str, columns[:12]))
    raise ValueError(
        f"Could not map {role} '{requested}' to a column in the input dataset. "
        f"Available columns sample: {sample}"
    )


def resolve_columns(
    columns: Iterable[str],
    requested_columns: List[str],
    role: str = "columns",
) -> Tuple[List[str], List[str]]:
    columns = list(columns)
    lookup = column_lookup(columns)
    resolved_columns = []
    missing = []
    for requested in requested_columns:
        requested = str(requested).strip()
        if not requested:
            continue
        if requested in columns:
            resolved_columns.append(requested)
            continue
        resolved = lookup.get(canonical_column_name(requested))
        if resolved is None:
            missing.append(requested)
        else:
            resolved_columns.append(resolved)
    if missing:
        print(
            f"-> WARNING: Skipping {len(missing)} {role} not found in the input dataset. "
            f"First skipped values: {', '.join(missing[:10])}."
        )
    return resolved_columns, missing


def resolve_feature_values(
    df_features: pd.DataFrame,
    requested_columns: List[str],
    role: str = "feature columns",
    fill_missing_with_zero: bool = True,
) -> np.ndarray:
    lookup = column_lookup(df_features.columns)
    values = []
    missing = []
    changed = 0
    for requested in requested_columns:
        requested = str(requested).strip()
        if not requested:
            continue
        if requested in df_features.columns:
            values.append(df_features[requested].values)
            continue
        resolved = lookup.get(canonical_column_name(requested))
        if resolved is None:
            missing.append(requested)
            if fill_missing_with_zero:
                values.append(np.zeros(len(df_features), dtype=np.float32))
        else:
            values.append(df_features[resolved].values)
            if resolved != requested:
                changed += 1
    if missing:
        action = "Using zero-filled placeholders" if fill_missing_with_zero else "Skipping them"
        print(
            f"-> WARNING: {len(missing)} {role} were not found in the input dataset. "
            f"{action} so processing can continue. "
            f"First missing values: {', '.join(missing[:10])}."
        )
    if changed:
        print(f"-> INFO: Mapped {changed} {role} to current dataset column names.")
    if not values:
        return np.empty((len(df_features), 0), dtype=np.float32)
    return np.column_stack(values)
