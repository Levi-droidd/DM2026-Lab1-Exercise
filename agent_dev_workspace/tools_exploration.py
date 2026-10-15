import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_1: Optional[int] = None,
        doc_index_2: Optional[int] = None,
        text_1: Optional[str] = None,
        text_2: Optional[str] = None,
    ) -> dict:
        """Compute the cosine similarity between two documents.

        Can compare two documents already in the document-term matrix by their
        row indices (doc_index_1, doc_index_2), or arbitrary raw text strings
        (text_1, text_2) vectorized using the fitted CountVectorizer artifact.
        Row indices take raw count vectors directly from session.feature_matrix.

        Args:
            doc_index_1: Index of the first document in session.feature_matrix.
            doc_index_2: Index of the second document in session.feature_matrix.
            text_1: Optional raw text for the first document.
            text_2: Optional raw text for the second document.

        Returns:
            A dict with result_id, doc_index_1, doc_index_2, and similarity, or error status.
        """
        # Resolve vector 1
        if text_1 is not None:
            vectorizer = session.artifacts.get("count_vectorizer")
            if vectorizer is None:
                return {"status": "error", "message": "CountVectorizer artifact not found in session.artifacts. Run build_dtm_tool first."}
            vec1 = vectorizer.transform([text_1])
        elif doc_index_1 is not None:
            if session.feature_matrix is None:
                return {"status": "error", "message": "No feature matrix found in session. Run build_dtm_tool first."}
            n_docs = session.feature_matrix.shape[0]
            if not (0 <= doc_index_1 < n_docs):
                return {"status": "error", "message": f"doc_index_1 {doc_index_1} is out of range [0, {n_docs - 1}]."}
            vec1 = session.feature_matrix[doc_index_1]
        else:
            return {"status": "error", "message": "Either doc_index_1 or text_1 must be provided."}

        # Resolve vector 2
        if text_2 is not None:
            vectorizer = session.artifacts.get("count_vectorizer")
            if vectorizer is None:
                return {"status": "error", "message": "CountVectorizer artifact not found in session.artifacts. Run build_dtm_tool first."}
            vec2 = vectorizer.transform([text_2])
        elif doc_index_2 is not None:
            if session.feature_matrix is None:
                return {"status": "error", "message": "No feature matrix found in session. Run build_dtm_tool first."}
            n_docs = session.feature_matrix.shape[0]
            if not (0 <= doc_index_2 < n_docs):
                return {"status": "error", "message": f"doc_index_2 {doc_index_2} is out of range [0, {n_docs - 1}]."}
            vec2 = session.feature_matrix[doc_index_2]
        else:
            return {"status": "error", "message": "Either doc_index_2 or text_2 must be provided."}

        # Check for empty / zero vectors to prevent NaN or division-by-zero
        norm1 = np.linalg.norm(vec1.toarray())
        norm2 = np.linalg.norm(vec2.toarray())

        if norm1 == 0.0 or norm2 == 0.0:
            sim_value = 0.0
        else:
            sim_mat = cosine_similarity(vec1, vec2)
            sim_value = float(sim_mat[0, 0])

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity": sim_value,
        }
        call_args = {
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "text_1": text_1,
            "text_2": text_2,
        }
        session.store_result("cosine_similarity_tool", call_args, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Compute the Pearson correlation matrix across the top 20 terms by variance in the global DTM.

        Finds the top 20 terms by variance (or all if fewer than 20), computes their
        Pearson correlation matrix using np.corrcoef, builds an annotated heatmap figure
        set on session.pending_figure, and stores the result.

        Returns:
            Dict containing result_id, term_names (sorted by descending variance),
            and the correlation_matrix as a nested list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature matrix or feature names found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        feature_names = session.feature_names

        # Compute sparse variance: E[X^2] - (E[X])^2 matching variance_filter_tool
        mean = X.mean(axis=0).A1
        mean_sq = X.power(2).mean(axis=0).A1
        variances = mean_sq - (mean ** 2)

        # Sort indices by variance descending
        sorted_indices = np.argsort(-variances)
        top_k = min(20, len(sorted_indices))
        selected_indices = sorted_indices[:top_k]

        selected_terms = [feature_names[i] for i in selected_indices]

        # Extract selected columns (dense array of shape: n_documents x top_k)
        selected_cols = X[:, selected_indices].toarray()

        # Compute Pearson correlation matrix across columns (features)
        # np.corrcoef computes correlation across rows by default, so set rowvar=False
        corr_matrix = np.corrcoef(selected_cols, rowvar=False)

        # Handle potential NaNs if any column had zero variance
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Ensure 2D array representation
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[float(corr_matrix)]])
        elif corr_matrix.ndim == 1:
            corr_matrix = corr_matrix.reshape(1, 1)

        # Build annotated heatmap and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(max(6, top_k * 0.5), max(5, top_k * 0.4)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            ax=ax,
            cbar=True,
        )
        ax.set_title("Feature Correlation Matrix (Top Terms by Variance)")
        plt.tight_layout()
        session.pending_figure = fig

        result_id = session.next_result_id("feat_corr")
        summary = {
            "result_id": result_id,
            "term_names": selected_terms,
            "correlation_matrix": corr_matrix.tolist(),
        }
        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
