# Agentic Pipeline Report

## 1. Data and Quality Check

I loaded 2,257 documents across 4 categories: soc.religion.christian (599), sci.med (594), comp.graphics (584), and alt.atheism (480) (`load_dataset_1`). The data quality is solid, with no missing values (`check_missing_3`) and no duplicated documents (`check_duplicates_4`). The document length distribution is heavily right-skewed (`describe_data_5`). As the box plot shows, soc.religion.christian has the highest median length and is generally longer than the others. In contrast, comp.graphics has the lowest median length but contains 4 extreme outliers exceeding 45,000 characters, which I suspect are encoded images. Because of this skew, the median is a more reliable metric than the mean when comparing text length across categories.

![Text length by category](../plots/彭祉錚_agentic-pipeline_plot_describe_data_5.png)

## 2. DTM and Term Frequencies

The document-term matrix has 35,788 distinct terms and a sparsity of 99.55% (`dtm_6`). The top 20 terms by frequency are almost entirely stopwords; for instance, "the" appears 29,825 times (`term_freq_7`). The only exception is "edu", which most likely comes from email addresses in the headers. The DTM heatmap of the first 20 alphabetically sorted terms, which are all numbers, shows the sparsity directly: only 2 of its 400 cells are non-zero (`dtm_heatmap_8`).

![DTM heatmap](../plots/彭祉錚_agentic-pipeline_plot_dtm_heatmap_8.png)

## 3. Filtering and Correlation

A variance filter with a 0.01 threshold kept 5,354 terms and removed 85% of the vocabulary, because most words appear in very few documents (`variance_filter_10`). A Pearson correlation filter against sci.med (threshold 0.1) kept 291 terms (`pearson_filter_11`). However, its bar chart reveals a data leakage issue: most of the top-correlated words (gordon, banks, geb, pitt, n3jxp, and words from a quote about skepticism) come from one prolific user's signature, and among the top 15 only "medicine" and "treatment" are actual medical terms. In other words, the filter partly detects who posts in sci.med rather than the topic itself. Spearman correlation kept 421 terms, 130 more than Pearson (`spearman_filter_12`), since its rank-based approach depends more on whether a word appears at all than on its exact count. The feature correlation heatmap (`feat_corr_13`) shows that stopwords are highly correlated with each other (0.75 to 0.94), which mainly reflects document length, while "edu" and "jpeg" have low correlations with the other terms.

![Pearson filter top terms](../plots/彭祉錚_agentic-pipeline_plot_turn68.png)

![Feature correlation matrix](../plots/彭祉錚_agentic-pipeline_plot_feat_corr_13.png)

## 4. Pattern Mining

For each category, I filtered the vocabulary by TF-IDF and mined the 20 highest-support patterns with Top-K (FAE) (`patterns_14` to `patterns_17`). The results are heavily dominated by header fields: in all 4 categories, "subject", "lines", "organization", "edu", and "writes" rank at the top. soc.religion.christian is the only category with a real topic word in its top 20: "god", with a support of 344 out of 599 documents. alt.atheism has the highest support for "writes" (81%), the word inserted when quoting a reply, which suggests many replies and debates. comp.graphics has the lowest support for "edu" (55%), suggesting a larger share of non-academic users.

## 5. Dimensionality Reduction, Similarity, and Conclusion

With PCA, PC1 explains 65.9% of the variance and PC2 explains 4.3% (`reduce_dim_18`). PC1 mainly captures document length, so the categories overlap heavily, and the distant outliers likely correspond to the very long documents noted earlier. t-SNE (random_state=42) tells a similar story (`reduce_dim_19`): only comp.graphics forms a somewhat distinct region, while the two religious categories overlap heavily. Cosine similarity on one pair from the same category (documents 0 and 1, both comp.graphics) was 0.353, and on one pair from different categories (documents 0 and 2, comp.graphics vs soc.religion.christian) was 0.340, a difference of only 0.013 (`cosine_sim_21`, `cosine_sim_22`). With only two pairs this is not conclusive, but it is consistent with the other results. In conclusion, the raw counts are largely dominated by stopwords, headers, and signatures, which mask the actual topic content. Next steps would be to remove headers and signatures when loading the data, drop stopwords from the DTM, and use TF-IDF weighting instead of raw counts.

![PCA](../plots/彭祉錚_agentic-pipeline_plot_reduce_dim_18.png)

![t-SNE](../plots/彭祉錚_agentic-pipeline_plot_reduce_dim_19.png)
