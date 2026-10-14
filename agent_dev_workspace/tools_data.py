import pandas as pd
import pathlib
from typing import Dict, Any, Optional
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes text in session.dataframe into unigrams using NLTK word_tokenize.

        Expects session.dataframe to contain a 'text' column. Lowercases text before
        tokenization. Stores tokens list in session.dataframe['unigrams'].

        Returns:
            Dict containing status and summary of tokenization.
        """
        df = session.dataframe
        if df is None or "text" not in df.columns:
            return {"status": "error", "message": "No dataset or 'text' column loaded in session."}

        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        from nltk.tokenize import word_tokenize

        tokens_list = []
        for text in df["text"]:
            if pd.isna(text) or str(text).strip() == "":
                tokens_list.append([])
            else:
                tokens_list.append(word_tokenize(str(text).lower()))

        session.dataframe["unigrams"] = tokens_list

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": len(df),
            "total_tokens": sum(len(t) for t in tokens_list),
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary,
        )

        return summary

    @tool
    def list_files_tool(subdirectory: Optional[str] = None) -> Dict[str, Any]:
        """Lists files and directories in the repository or a specified subdirectory.

        Args:
            subdirectory: Optional relative path within the repository (e.g., 'newdataset').

        Returns:
            Dict containing lists of 'directories' and 'files' found.
        """
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        
        if subdirectory:
            sub_path = pathlib.Path(subdirectory)
            if sub_path.is_absolute() or ".." in sub_path.parts:
                return {"status": "error", "message": "Access denied: Path traversal or absolute paths are not allowed."}
            target_path = (repo_root / sub_path).resolve()
            if not target_path.is_relative_to(repo_root):
                return {"status": "error", "message": "Access denied: Path outside repository root."}
        else:
            target_path = repo_root

        if not target_path.exists() or not target_path.is_dir():
            return {"status": "error", "message": f"Directory not found: {subdirectory}"}

        directories = []
        files = []
        for item in target_path.iterdir():
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory or "",
            "directories": directories,
            "files": files,
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary,
        )

        return summary

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

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Calculates text length statistics overall and per category, and builds a boxplot.

        Returns:
            Dict containing overall and per-category text length summary statistics.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        df = session.dataframe
        if df is None or "text" not in df.columns:
            return {"status": "error", "message": "No dataset or 'text' column loaded in session."}

        # Safe character length calculation handling NaNs
        text_length = df["text"].fillna("").astype(str).apply(len)

        # Overall summary stats
        overall_stats = text_length.describe().to_dict()

        # Per category summary stats
        cat_col = "category_name" if "category_name" in df.columns else "category"
        per_category_stats = {}
        if cat_col in df.columns:
            for cat, group in df.groupby(cat_col):
                per_category_stats[str(cat)] = group["text"].fillna("").astype(str).apply(len).describe().to_dict()

        # Build boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        if cat_col in df.columns:
            plot_df = pd.DataFrame({"text_length": text_length, cat_col: df[cat_col]})
            sns.boxplot(data=plot_df, x=cat_col, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(y=text_length, ax=ax)
            ax.set_title("Text Length Distribution")
        ax.set_ylabel("Text Length (chars)")

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall": overall_stats,
            "per_category": per_category_stats,
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary,
        )

        return summary

    @tool
    def inspect_data_tool(n: int = 3) -> Dict[str, Any]:
        """Inspects session.dataframe returning total row count, column names, and the first n rows.

        Args:
            n: Number of rows to return (default 3). Text values longer than 100 characters are truncated with '...'.

        Returns:
            Dict containing total_rows, columns, and preview_rows.
        """
        df = session.dataframe
        if df is None:
            return {"status": "error", "message": "No dataset loaded in session."}

        total_rows = int(len(df))
        columns = list(df.columns)
        
        head_df = df.head(n).copy()
        for col in head_df.columns:
            head_df[col] = head_df[col].apply(
                lambda val: (str(val)[:100] + "...") if isinstance(val, str) and len(val) > 100 else val
            )
        
        preview_rows = head_df.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "columns": columns,
            "preview_rows": preview_rows,
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n": n},
            summary=summary,
        )

        return summary

    return [tokenize_tool, list_files_tool, check_missing_tool, check_duplicates_tool, sample_data_tool, describe_data_tool, inspect_data_tool]
