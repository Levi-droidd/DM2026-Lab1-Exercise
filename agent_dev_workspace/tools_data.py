import pandas as pd
from typing import Dict, Any, Optional
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Checks for missing, null, or empty text values in session.dataframe.

        Identifies both pandas NaN/None values and empty/whitespace-only strings
        in the 'text' column.

        Returns:
            Dict containing summary statistics about missing entries.
        """
        df = session.dataframe
        if df is None or "text" not in df.columns:
            return {"status": "error", "message": "No dataset or 'text' column loaded in session."}

        text_col = df["text"]
        
        # Catch pandas nulls (NaN, None)
        is_null = text_col.isna()
        
        # Catch empty strings or whitespace-only strings
        is_empty_str = text_col.astype(str).str.strip() == ""
        
        # Combine missing flags
        missing_mask = is_null | is_empty_str
        
        missing_count = int(missing_mask.sum())
        total_rows = len(df)
        missing_pct = float(missing_count / total_rows * 100) if total_rows > 0 else 0.0
        
        all_missing_indices = df.index[missing_mask].tolist()
        missing_indices_truncated = len(all_missing_indices) > 20
        missing_indices = all_missing_indices[:20]

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "missing_indices": missing_indices,
            "missing_indices_truncated": missing_indices_truncated,
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={},
            summary=summary,
        )

        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Checks for (and optionally drops) duplicate documents in session.dataframe.

        Args:
            drop: If True, drops all copies of duplicate rows using keep=False.

        Returns:
            Dict containing count of duplicate entries and row counts.
        """
        df = session.dataframe
        if df is None or "text" not in df.columns:
            return {"status": "error", "message": "No dataset or 'text' column loaded in session."}

        # Count extra duplicate rows using keep='first'
        dup_mask = df.duplicated(subset=["text"], keep="first")
        duplicate_count = int(dup_mask.sum())
        
        all_dup_indices = df.index[dup_mask].tolist()
        duplicate_indices_truncated = len(all_dup_indices) > 20
        duplicate_indices = all_dup_indices[:20]

        if drop:
            # Drop every copy of duplicate rows using keep=False
            session.dataframe.drop_duplicates(subset=["text"], keep=False, inplace=True)

        rows_remaining = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "duplicate_indices_truncated": duplicate_indices_truncated,
            "dropped": drop,
            "rows_remaining": rows_remaining,
        }

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"drop": drop},
            summary=summary,
        )

        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Dict[str, Any]:
        """Subsamples rows from session.dataframe and replaces the active DataFrame.

        Args:
            n: Number of rows to sample. Must specify exactly one of n or frac.
            frac: Fraction of rows to sample (0.0 to 1.0). Must specify exactly one of n or frac.
            random_state: Seed for random sampling reproducibility.

        Returns:
            Dict summarizing the subsampling operation.
        """
        df = session.dataframe
        if df is None:
            return {"status": "error", "message": "No dataset loaded in session."}

        if (n is None and frac is None) or (n is not None and frac is not None):
            return {
                "status": "error",
                "message": "Specify exactly one of 'n' or 'frac'."
            }

        sampled_df = df.sample(n=n, frac=frac, random_state=random_state)
        session.dataframe = sampled_df

        sampled_rows = len(sampled_df)
        all_sample_indices = sampled_df.index.tolist()
        sample_indices_truncated = len(all_sample_indices) > 20
        sample_indices = all_sample_indices[:20]

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "sampled_rows": sampled_rows,
            "sample_indices": sample_indices,
            "sample_indices_truncated": sample_indices_truncated,
            "random_state": random_state,
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "frac": frac, "random_state": random_state},
            summary=summary,
        )

        return summary

    return [check_missing_tool, check_duplicates_tool, sample_data_tool]
