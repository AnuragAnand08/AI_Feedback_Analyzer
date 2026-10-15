"""Loading, validating and cleaning feedback data."""
import pandas as pd

FEEDBACK_COLUMN_CANDIDATES = [
    "feedback", "comment", "comments", "review", "reviews",
    "response", "responses", "remarks", "text", "suggestion",
]
STANDARD_COLUMNS = ["feedback", "course", "date", "source"]
MISSING_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-"}
MIN_LENGTH = 3


class DataLoadError(Exception):
    """Raised when a file cannot be used; the message is safe to show to the user."""


def empty_feedback_df():
    return pd.DataFrame(columns=["id"] + STANDARD_COLUMNS)


def _read_csv(source):
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding, on_bad_lines="skip")
        except pd.errors.EmptyDataError:
            raise DataLoadError("The CSV file is empty.")
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            raise DataLoadError(
                f"The file could not be read as a CSV ({type(exc).__name__}). "
                "Please upload a valid .csv file."
            )
    raise DataLoadError("The file encoding is not supported. Save it as UTF-8 CSV.")


def find_feedback_column(df):
    """Find the column holding the feedback text."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for candidate in FEEDBACK_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    # Fallback: the text column with the longest average length
    best, best_len = None, 0
    for col in df.columns:
        if df[col].dtype == object:
            avg = df[col].dropna().astype(str).str.len().mean()
            if avg == avg and avg > best_len:  # avg == avg filters out NaN
                best, best_len = col, avg
    return best if best_len >= 15 else None


def clean_feedback_df(df):
    """Standardise columns, drop missing/short rows and duplicates.

    Returns (clean_dataframe, list_of_warning_messages).
    """
    warnings = []
    if df is None or df.empty or "feedback" not in df.columns:
        return empty_feedback_df(), ["No feedback rows were found."]

    out = df.copy()
    for col in ("course", "date", "source"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)

    out["feedback"] = (
        out["feedback"].fillna("").astype(str)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    invalid = out["feedback"].str.lower().isin(MISSING_TOKENS) | (out["feedback"].str.len() < MIN_LENGTH)
    if int(invalid.sum()):
        warnings.append(f"Skipped {int(invalid.sum())} row(s) with missing or too-short feedback.")
    out = out[~invalid]

    key = out["feedback"].str.lower().str.replace(r"[^\w\s]", "", regex=True)
    duplicated = key.duplicated()
    if int(duplicated.sum()):
        warnings.append(f"Removed {int(duplicated.sum())} duplicate feedback entr(ies).")
    out = out[~duplicated]

    if out.empty:
        return empty_feedback_df(), warnings + ["No valid feedback remained after cleaning."]

    out = out[STANDARD_COLUMNS].reset_index(drop=True)
    out.insert(0, "id", range(1, len(out) + 1))
    return out, warnings


def load_csv(source, source_name="Uploaded CSV"):
    """Read a CSV (path or file-like object). Returns (clean_df, warnings)."""
    raw = _read_csv(source)
    if raw.empty:
        raise DataLoadError("The CSV file contains no rows.")
    raw = raw.rename(columns=lambda c: str(c).strip())

    column = find_feedback_column(raw)
    if column is None:
        raise DataLoadError(
            "Could not find a feedback column. Rename the column containing the "
            "comments to 'feedback' and upload again."
        )

    lowered = {c.lower(): c for c in raw.columns}
    data = pd.DataFrame({"feedback": raw[column]})
    for optional in ("course", "date"):
        data[optional] = raw[lowered[optional]] if optional in lowered else ""
    data["source"] = source_name

    cleaned, warnings = clean_feedback_df(data)
    if cleaned.empty:
        raise DataLoadError("; ".join(warnings) or "No valid feedback found in the file.")
    return cleaned, warnings