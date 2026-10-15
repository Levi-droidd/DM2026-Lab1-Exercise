import os
import tempfile
import numpy as np
import pandas as pd
from typing import Literal, Optional, Dict, Any, List
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import PAMI.frequentPattern.basic.FPGrowth as FPGrowth
import PAMI.frequentPattern.topk.FAE as FAE
import PAMI.frequentPattern.maximal.MaxFPGrowth as MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> dict:
        """Mine frequent word patterns for a specific category using PAMI.

        Constructs a category-local document-term matrix with stop_words='english',
        filters the vocabulary according to filtering_method, converts surviving terms
        to a transactional database, and runs the requested mining algorithm.

        Args:
            category_name: Name of the category to filter documents by.
            filtering_method: Filtering method ("variance", "tfidf", or "term_frequency").
            algorithm: PAMI mining algorithm ("fpgrowth", "topk", or "maxfpgrowth"). Default "fpgrowth".
            min_sup: Minimum support count (required for "fpgrowth" and "maxfpgrowth").
            k: Top-K count (required for "topk").

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm,
            n_patterns, and the first 20 patterns sorted by support descending.
        """
        # Parameter validation
        if algorithm in ("fpgrowth", "maxfpgrowth") and min_sup is None:
            return {"status": "error", "message": f"min_sup is required for algorithm '{algorithm}'."}
        if algorithm == "topk" and k is None:
            return {"status": "error", "message": "k is required for algorithm 'topk'."}

        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe loaded in session. Run load_dataset_tool first."}

        df = session.dataframe
        if "category_name" in df.columns:
            cat_mask = (df["category_name"] == category_name)
        elif "label" in df.columns:
            cat_mask = (df["label"] == category_name)
        else:
            cat_mask = (df["category"] == category_name)

        cat_df = df[cat_mask]
        if len(cat_df) == 0:
            return {"status": "error", "message": f"No documents found for category '{category_name}'."}

        # Step 1: Category-local CountVectorizer with English stop words
        vectorizer = CountVectorizer(stop_words="english")
        try:
            local_dtm = vectorizer.fit_transform(cat_df["text"].fillna(""))
        except ValueError:
            # Empty vocabulary after stop words
            local_dtm = None

        feature_names = vectorizer.get_feature_names_out() if local_dtm is not None else []
        n_features = len(feature_names)

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        call_args = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }

        # Step 2: Vocabulary filtering
        kept_indices = []
        if n_features > 0:
            X_dense = local_dtm.toarray()
            if filtering_method == "variance":
                # Variance across documents: keep (variance > p5) and (variance < p95)
                # Middle 90% strictly between 5th and 95th percentiles
                variances = np.var(X_dense, axis=0)
                p5 = np.percentile(variances, 5)
                p95 = np.percentile(variances, 95)
                kept_indices = [i for i, v in enumerate(variances) if p5 < v < p95]

            elif filtering_method == "term_frequency":
                # Column sums: keep (sum > p5) and (sum < p95)
                # Middle 90% strictly between 5th and 95th percentiles
                col_sums = np.sum(X_dense, axis=0)
                p5 = np.percentile(col_sums, 5)
                p95 = np.percentile(col_sums, 95)
                kept_indices = [i for i, s in enumerate(col_sums) if p5 < s < p95]

            elif filtering_method == "tfidf":
                # Mean TF-IDF across documents: keep mean_tfidf >= p20 (top 80%)
                tfidf_mat = TfidfTransformer().fit_transform(local_dtm).toarray()
                mean_tfidf = np.mean(tfidf_mat, axis=0)
                p20 = np.percentile(mean_tfidf, 20)
                kept_indices = [i for i, m in enumerate(mean_tfidf) if m >= p20]

        # Step 3: Handle zero terms survived filtering cleanly
        if len(kept_indices) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.artifacts[artifact_key] = []
            session.store_result("mine_patterns_tool", call_args, summary)
            return summary

        kept_terms = [feature_names[i] for i in kept_indices]
        filtered_counts = X_dense[:, kept_indices]

        # Step 4: Write transactions to a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".txt", text=True)
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                for row in filtered_counts:
                    transaction = [term for term, count in zip(kept_terms, row) if count >= 1]
                    if transaction:
                        f.write("\t".join(transaction) + "\n")

            # Run chosen PAMI mining algorithm
            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(temp_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = FAE.FAE(temp_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(temp_path, minSup=min_sup, sep="\t")

            # Use mine() if available, fallback to startMine()
            if hasattr(miner, "mine"):
                miner.mine()
            else:
                miner.startMine()

            # Retrieve patterns
            patterns_dict = miner.getPatterns()
            patterns = []
            if patterns_dict:
                for pat_key, sup in patterns_dict.items():
                    # PAMI can return pattern keys as tuple/list or delimiter-separated str
                    if isinstance(pat_key, (tuple, list)):
                        pat_terms = [str(t) for t in pat_key]
                    else:
                        pat_str = str(pat_key).strip()
                        pat_terms = pat_str.split("\t") if "\t" in pat_str else pat_str.split()
                    patterns.append({
                        "pattern": pat_terms,
                        "support": int(sup),
                    })
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Sort patterns by support descending
        patterns.sort(key=lambda x: x["support"], reverse=True)

        # Step 5 & 6: Store full list in artifacts, cap summary at 20
        session.artifacts[artifact_key] = patterns
        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns),
            "patterns": patterns[:20],
        }
        session.store_result("mine_patterns_tool", call_args, summary)
        return summary

    return [mine_patterns_tool]
