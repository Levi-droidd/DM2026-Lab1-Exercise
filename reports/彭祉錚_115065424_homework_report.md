# Homework Report: Reddit Stock Sentiment

## 1. Data and Quality Check

The dataset consists of 847 posts in three sentiment classes: negative (-1.0), neutral (0.0), and positive (1.0) (`load_dataset_1`). There were no missing values (`check_missing_3`). However, there were 23 duplicate texts, which appear to be automated template posts such as daily price updates (`check_duplicates_4`). I dropped them with `keep=False`, which removes every copy, so 34 rows were removed and 813 posts remained (`check_duplicates_5`). The removed rows were almost all neutral: the neutral class went from 423 to 391, while negative stayed at 315 (`binarize_labels_19`). I also noticed that a post saying "Seeing lots of red in the ticker" is labeled neutral, even though "red" means falling prices in finance, which suggests the labels may be noisy for finance slang.

## 2. Post Length and Document-Term Matrix

The posts are very short, averaging about 148 characters, roughly 1/15 the length of the newsgroup articles (`describe_data_6`). Positive posts have the highest mean length (166), but negative posts have the highest median (96 vs 81), so the positive mean is pulled up by a few long outliers; as with the newsgroups, the median is the more reliable comparison here. The document-term matrix has 4,313 terms, about 1/8 of the newsgroup vocabulary, yet its sparsity is almost identical at 99.52% (`dtm_7`, `dtm_heatmap_9`). The top 20 terms are all stopwords, but unlike the newsgroups there is no "edu", since Reddit posts have no email headers (`term_freq_8`).

![Text length by sentiment](../plots/彭祉錚_homework_plot_describe_data_6.png)

![DTM heatmap](../plots/彭祉錚_homework_plot_dtm_heatmap_9.png)

## 3. Feature Filtering and Correlation

A variance filter with a 0.01 threshold kept 490 of 4,313 terms, removing about 89% (`variance_filter_10`). A Pearson filter for the negative class (threshold 0.1) kept only 18 terms, including "trump", "china", "economy", and several profanities; "great" also appeared, possibly from sarcastic use (`pearson_filter_11`). The highest correlation was only r = 0.176, much lower than the 0.38 for sci.med in the newsgroups, which suggests that sentiment is a much weaker signal in word counts than topic. Spearman kept 21 terms, mostly adding stopwords such as "the" and "and" (`spearman_filter_12`). The feature correlation matrix also shows weaker correlations than the newsgroups, with a maximum of 0.69 between "the" and "in", while "will" and "be" (0.59) reflect the common phrase "will be" (`feat_corr_13`).

![Pearson filter top terms](../plots/彭祉錚_homework_plot_turn70.png)

![Feature correlation matrix](../plots/彭祉錚_homework_plot_feat_corr_13.png)

## 4. Pattern Mining

For each class, I filtered the vocabulary by TF-IDF and mined the 20 highest-support patterns with Top-K (`patterns_14` to `patterns_16`). Unlike the newsgroups, the results were not affected by email headers, so real topic words appeared directly. "trump" and "china" rank near the top for both negative and neutral posts but do not appear in the positive top 20, while positive posts include investment words like "technology", "silver", and "hold". "tariffs" appears in all three classes. All patterns were single words, likely because the posts are too short for word pairs to co-occur often.

## 5. Dimensionality Reduction and Conclusion

With PCA, PC1 explains only 26.7% of the variance, compared with 65.9% for the newsgroups, and the three sentiment classes are completely mixed (`reduce_dim_17`). t-SNE spreads the points more evenly, but the classes still do not separate at all (`reduce_dim_18`). In conclusion, a bag-of-words representation is better at separating topics than sentiment. All three classes discuss the same topics (Trump, China, tariffs), and the real differences lie in tone, sarcasm, and finance slang, which raw word counts cannot capture. Cosine similarity was not run because the API quota ran out.

![PCA](../plots/彭祉錚_homework_plot_reduce_dim_17.png)

![t-SNE](../plots/彭祉錚_homework_plot_reduce_dim_18.png)