import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool
from agent_pipeline.session_state import ResultRecord

def make_tools(session):
    
    def _compute_correlation_filter(target_class: str, method: str, threshold: float) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"status": "error", "message": "No feature matrix or labels loaded in session."}, None

        X = session.feature_matrix
        feature_names = session.feature_names
        labels = np.array(session.labels)

        # One-vs-rest binary target: 1 if label == target_class else 0
        y = (labels == target_class).astype(int)

        correlations = []
        kept_features = []
        removed_features = []

        for idx, name in enumerate(feature_names):
            # Extract single column efficiently without converting entire matrix to dense
            col = X[:, idx].toarray().ravel()
            # Handle zero-variance terms (constant columns) so they don't return NaN
            if np.all(col == col[0]):
                r = 0.0
            else:
                if method == "pearson":
                    r, _ = pearsonr(col, y)
                else:
                    r, _ = spearmanr(col, y)
                if np.isnan(r):
                    r = 0.0

            r_float = float(r)
            correlations.append({"term": name, f"{method}_r": r_float})

            if abs(r_float) >= threshold:
                kept_features.append(name)
            else:
                removed_features.append(name)

        col_name = f"{method}_r"
        full_report_df = pd.DataFrame(correlations)
        full_report_df = full_report_df.sort_values(by=col_name, ascending=False).reset_index(drop=True)

        summary = {
            "target_class": target_class,
            "threshold": threshold,
            "total_features": len(feature_names),
            "kept_count": len(kept_features),
            "removed_count": len(removed_features),
            "kept_features": kept_features[:20],
            "removed_features": removed_features[:20],
        }

        return summary, full_report_df

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Filters out near-constant terms from session.feature_matrix based on variance.

        Args:
            threshold: Minimum variance required for a term to be kept (default 0.0).

        Returns:
            Dict containing filtering results, kept/removed counts, and summary statistics.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature matrix loaded in session."}

        X = session.feature_matrix
        feature_names = session.feature_names

        mean = X.mean(axis=0).A1
        mean_sq = X.power(2).mean(axis=0).A1
        variances = mean_sq - (mean ** 2)

        feature_variances = []
        for name, var in zip(feature_names, variances):
            var_float = float(var)
            feature_variances.append({"term": name, "variance": var_float})

        kept_all = [name for name, var in zip(feature_names, variances) if var >= threshold]
        removed_all = [name for name, var in zip(feature_names, variances) if var < threshold]

        full_report_df = pd.DataFrame(feature_variances)
        full_report_df = full_report_df.sort_values(by="variance", ascending=False).reset_index(drop=True)

        summary = {
            "threshold": threshold,
            "total_features": len(feature_names),
            "kept_count": len(kept_all),
            "removed_count": len(removed_all),
            "kept_features": kept_all[:20],
            "removed_features": removed_all[:20],
        }

        result_id = session.next_result_id("variance_filter")
        summary["result_id"] = result_id

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=full_report_df
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Filters terms based on Pearson correlation with a target class (one-vs-rest).

        Args:
            target_class: The target class label for binary correlation.
            threshold: Minimum absolute Pearson correlation required to keep a term (default 0.0).

        Returns:
            Dict containing filtering results, kept/removed counts, and summary statistics.
        """
        summary, full_report_df = _compute_correlation_filter(target_class, "pearson", threshold)
        if summary and "status" in summary and summary["status"] == "error":
            return summary

        result_id = session.next_result_id("pearson_filter")
        summary["result_id"] = result_id

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=full_report_df
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> Dict[str, Any]:
        """Filters terms based on Spearman correlation with a target class (one-vs-rest).

        Args:
            target_class: The target class label for binary correlation.
            threshold: Minimum absolute Spearman correlation required to keep a term (default 0.0).

        Returns:
            Dict containing filtering results, kept/removed counts, and summary statistics.
        """
        summary, full_report_df = _compute_correlation_filter(target_class, "spearman", threshold)
        if summary and "status" in summary and summary["status"] == "error":
            return summary

        result_id = session.next_result_id("spearman_filter")
        summary["result_id"] = result_id

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=full_report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
