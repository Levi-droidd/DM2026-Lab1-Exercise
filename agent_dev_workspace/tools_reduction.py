import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal, Optional, Dict, Any, List
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MultiLabelBinarizer
import umap


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = None,
    ) -> dict:
        """Reduce dimensions of document-term matrix using PCA, t-SNE, or UMAP, and plot a 2D scatter plot.

        Every run builds a scatter plot of the 2D coordinates colored by category label,
        attaches it to session.pending_figure, and stores the results.

        Args:
            method: Dimensionality reduction technique ("pca", "tsne", or "umap"). Default "pca".
            n_components: Number of target dimensions (default 2).
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors parameter for UMAP (default 15).
            random_state: Optional random seed for reproducible runs (t-SNE and UMAP).

        Returns:
            Dict containing result_id, method, n_components, coordinates (first 20),
            and explained_variance_ratio (if method is 'pca'), or error status.
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No feature matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix.toarray()
        n_samples = X.shape[0]

        if n_samples < 2:
            return {"status": "error", "message": "At least 2 documents are required for dimensionality reduction."}

        explained_variance_ratio: Optional[List[float]] = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = [float(x) for x in reducer.explained_variance_ratio_]
        elif method == "tsne":
            if perplexity >= n_samples:
                return {
                    "status": "error",
                    "message": f"perplexity ({perplexity}) must be less than n_samples ({n_samples}).",
                }
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords = reducer.fit_transform(X)
        else:
            return {"status": "error", "message": f"Unsupported method: {method}"}

        # Resolve labels for scatter plot coloring
        labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].tolist()
        if labels is None and session.labels is not None:
            labels = list(session.labels)
        if labels is None:
            labels = ["document"] * n_samples

        # Build 2D scatter plot and attach to session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_df = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1] if n_components > 1 else np.zeros(n_samples),
            "Category": labels,
        })
        sns.scatterplot(
            data=plot_df,
            x="Dim 1",
            y="Dim 2",
            hue="Category",
            style="Category",
            s=80,
            ax=ax,
        )
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        plt.tight_layout()
        session.pending_figure = fig

        # Save artifacts and prepare full report DataFrame
        session.artifacts["reduced_coordinates"] = coords
        report_df = pd.DataFrame(
            coords,
            columns=[f"dim_{i+1}" for i in range(coords.shape[1])]
        )
        report_df["category"] = labels

        result_id = session.next_result_id("reduce_dim")
        summary: Dict[str, Any] = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": coords[:20].tolist(),
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        call_args = {
            "method": method,
            "n_components": n_components,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
            "random_state": random_state,
        }
        session.store_result("reduce_dimensions_tool", call_args, summary, full_report=report_df)
        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encode document category labels using MultiLabelBinarizer.

        Wraps each category string in a list to prevent character splitting, fits
        MultiLabelBinarizer, and stores both the fitted binarizer and the binary matrix Y
        in session.artifacts.

        Returns:
            Dict containing result_id, classes, shape of the one-hot encoded matrix,
            and document counts per class.
        """
        if session.dataframe is None and session.labels is None:
            return {"status": "error", "message": "No dataset loaded in session. Run load_dataset_tool first."}

        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].tolist()
        elif session.labels is not None:
            labels = list(session.labels)
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            labels = session.dataframe["category"].tolist()

        if not labels:
            return {"status": "error", "message": "No category labels found to binarize."}

        # Wrap each label in a list so MultiLabelBinarizer doesn't split characters
        wrapped_labels = [[str(lbl)] for lbl in labels]

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(wrapped_labels)

        # Store artifacts
        session.artifacts["label_binarizer"] = mlb
        session.artifacts["binarized_labels"] = Y

        classes_list = [str(cls) for cls in mlb.classes_]
        class_counts = {cls: int(count) for cls, count in zip(classes_list, Y.sum(axis=0))}

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": classes_list,
            "shape": list(Y.shape),
            "counts_per_class": class_counts,
        }
        session.store_result("binarize_labels_tool", {}, summary)
        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
