from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session: Any) -> List[Any]:
    """Factory function that creates tools for DTM operations, bound to session."""

    @tool
    def build_dtm_tool(
        text_column: str = "text",
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        max_df: float = 1.0,
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from the text column in session.dataframe.

        Fits a CountVectorizer on session.dataframe[text_column], stores the sparse feature
        matrix, feature names, and fitted vectorizer on session, updates document labels if present,
        and returns summary statistics including matrix sparsity.

        Args:
            text_column: The column in session.dataframe containing document text.
            ngram_range: Lower and upper boundary of the range of n-values for different n-grams.
                Passed as a two-element list [min_n, max_n] to satisfy API schema requirements.
            min_df: Minimum document frequency threshold for CountVectorizer.
            max_df: Maximum document frequency threshold ratio or count for CountVectorizer.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}
        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in dataframe."}

        # Convert empty/NaN strings to empty text to avoid vectorizer errors
        texts = session.dataframe[text_column].fillna("").astype(str).tolist()

        # Convert List[int] to tuple for CountVectorizer
        tuple_ngram = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=tuple_ngram,
            min_df=min_df,
            max_df=max_df,
        )

        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = X
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        n_documents, n_terms = X.shape
        non_zero = int(X.nnz)
        total_elements = n_documents * n_terms
        sparsity_pct = round(100.0 * (1.0 - non_zero / total_elements), 4) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        args = {
            "text_column": text_column,
            "ngram_range": ngram_range,
            "min_df": min_df,
            "max_df": max_df,
        }
        # Keep summary lightweight: store a small sample of feature names rather than all 35k+
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
            "vocabulary_sample": feature_names[:10],
        }

        session.store_result("build_dtm_tool", args, summary)
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in session.feature_matrix.

        Sums counts for each term across all rows in the DTM, ranks terms by total frequency,
        and returns the top terms along with storing a full DataFrame report for visualization.

        Args:
            top_n: Number of top frequent terms to return in the summary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not built. Call build_dtm_tool first."}

        # Sum total frequency across all documents (axis=0)
        counts = np.asarray(session.feature_matrix.sum(axis=0)).ravel()

        # Build DataFrame and sort descending
        df_freq = pd.DataFrame({
            "term": session.feature_names,
            "frequency": counts
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        # Prepare top_n slice for summary
        top_df = df_freq.head(top_n)
        term_freq_list = top_df.to_dict(orient="records")

        result_id = session.next_result_id("term_freq")
        args = {"top_n": top_n}
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "top_n_requested": top_n,
            "term_frequencies": term_freq_list,
        }

        # Store full_report DataFrame formatted as (term, frequency) so visualize_result_tool can plot it
        session.store_result("term_frequency_tool", args, summary, full_report=df_freq)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap of a positional slice of the raw Document-Term Matrix (DTM).

        Slices the first n_documents rows and first n_terms columns of session.feature_matrix,
        creates a heatmap visualization set on session.pending_figure, and returns the matrix slice.

        Args:
            n_terms: Number of terms (columns) to slice from the beginning of feature_names.
            n_documents: Number of documents (rows) to slice from the beginning of session.dataframe.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not built. Call build_dtm_tool first."}

        actual_n_docs = min(n_documents, session.feature_matrix.shape[0])
        actual_n_terms = min(n_terms, session.feature_matrix.shape[1])

        # Extract positional submatrix slice and term subset
        submatrix = session.feature_matrix[:actual_n_docs, :actual_n_terms].toarray()
        terms_subset = session.feature_names[:actual_n_terms]
        doc_indices = [f"Doc {i}" for i in range(actual_n_docs)]

        # Generate plot
        fig, ax = plt.subplots(figsize=(max(6, actual_n_terms * 0.8), max(4, actual_n_docs * 0.5)))
        sns.heatmap(
            submatrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=terms_subset,
            yticklabels=doc_indices,
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"DTM Heatmap (First {actual_n_docs} Docs x {actual_n_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set pending figure on session
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        args = {"n_terms": n_terms, "n_documents": n_documents}
        summary = {
            "result_id": result_id,
            "n_documents_sliced": actual_n_docs,
            "n_terms_sliced": actual_n_terms,
            "terms": terms_subset,
            "documents": doc_indices,
            "matrix_slice": submatrix.tolist(),
        }

        session.store_result("dtm_heatmap_tool", args, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
