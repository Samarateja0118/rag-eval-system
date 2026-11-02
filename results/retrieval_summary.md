| run | chunking | retriever | filter | R@1 | R@5 | R@10 | R@20 | MRR@10 | doc hit@5 | ctx tokens@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| chunk-page__bm25__auto_company | page | bm25 | auto_company | 0.107 | 0.197 | 0.303 | 0.390 | 0.159 | 0.920 | 4021 |
| chunk-page__dense__auto_company | page | dense | auto_company | 0.123 | 0.327 | 0.448 | 0.614 | 0.223 | 0.880 | 3134 |
| chunk-page__rrf__auto_company | page | rrf | auto_company | 0.107 | 0.327 | 0.480 | 0.630 | 0.229 | 0.953 | 3531 |
| chunk-tok128__bm25__auto_company | tok128 | bm25 | auto_company | 0.130 | 0.207 | 0.287 | 0.403 | 0.190 | 0.853 | 636 |
| chunk-tok128__dense__auto_company | tok128 | dense | auto_company | 0.108 | 0.306 | 0.429 | 0.572 | 0.209 | 0.913 | 634 |
| chunk-tok128__rrf__auto_company | tok128 | rrf | auto_company | 0.123 | 0.314 | 0.474 | 0.642 | 0.232 | 0.927 | 636 |
| chunk-tok256__bm25__auto_company | tok256 | bm25 | auto_company | 0.120 | 0.193 | 0.303 | 0.427 | 0.177 | 0.887 | 1275 |
| chunk-tok256__dense__auto_company | tok256 | dense | auto_company | 0.110 | 0.321 | 0.521 | 0.664 | 0.217 | 0.873 | 1258 |
| chunk-tok256__rrf__auto_company | tok256 | rrf | auto_company | 0.140 | 0.340 | 0.503 | 0.647 | 0.261 | 0.873 | 1273 |
| chunk-tok512__bm25__auto_company | tok512 | bm25 | auto_company | 0.140 | 0.233 | 0.297 | 0.433 | 0.187 | 0.860 | 2556 |
| chunk-tok512__dense__auto_company | tok512 | dense | auto_company | 0.090 | 0.342 | 0.496 | 0.659 | 0.214 | 0.887 | 2537 |
| chunk-tok512__rrf__auto_company | tok512 | rrf | auto_company | 0.153 | 0.320 | 0.447 | 0.630 | 0.246 | 0.880 | 2555 |
| chunk-sent256__bm25__auto_company | sent256 | bm25 | auto_company | 0.110 | 0.203 | 0.260 | 0.370 | 0.162 | 0.867 | 1117 |
| chunk-sent256__dense__auto_company | sent256 | dense | auto_company | 0.100 | 0.303 | 0.403 | 0.547 | 0.196 | 0.867 | 919 |
| chunk-sent256__rrf__auto_company | sent256 | rrf | auto_company | 0.103 | 0.293 | 0.407 | 0.537 | 0.204 | 0.900 | 1057 |
| chunk-tok256__bm25__none | tok256 | bm25 | none | 0.090 | 0.150 | 0.190 | 0.283 | 0.126 | 0.513 | 1278 |
| chunk-tok256__dense__none | tok256 | dense | none | 0.110 | 0.304 | 0.478 | 0.638 | 0.204 | 0.847 | 1258 |
| chunk-tok256__rrf__none | tok256 | rrf | none | 0.127 | 0.303 | 0.403 | 0.568 | 0.214 | 0.787 | 1273 |
| chunk-tok256__bm25__auto_company_year | tok256 | bm25 | auto_company_year | 0.123 | 0.263 | 0.403 | 0.489 | 0.206 | 0.920 | 1266 |
| chunk-tok256__dense__auto_company_year | tok256 | dense | auto_company_year | 0.127 | 0.399 | 0.574 | 0.691 | 0.264 | 0.900 | 1255 |
| chunk-tok256__rrf__auto_company_year | tok256 | rrf | auto_company_year | 0.160 | 0.393 | 0.560 | 0.691 | 0.301 | 0.907 | 1264 |
| chunk-tok256__bm25__oracle_doc | tok256 | bm25 | oracle_doc | 0.170 | 0.407 | 0.520 | 0.652 | 0.288 | 1.000 | 1276 |
| chunk-tok256__dense__oracle_doc | tok256 | dense | oracle_doc | 0.180 | 0.541 | 0.714 | 0.833 | 0.361 | 1.000 | 1266 |
| chunk-tok256__rrf__oracle_doc | tok256 | rrf | oracle_doc | 0.261 | 0.584 | 0.683 | 0.803 | 0.425 | 1.000 | 1274 |
| chunk-tok256__weighted0.3__auto_company | tok256 | weighted a=0.3 | auto_company | 0.130 | 0.257 | 0.387 | 0.497 | 0.211 | 0.893 | 1275 |
| chunk-tok256__weighted0.5__auto_company | tok256 | weighted a=0.5 | auto_company | 0.147 | 0.307 | 0.440 | 0.594 | 0.240 | 0.900 | 1273 |
| chunk-tok256__weighted0.7__auto_company | tok256 | weighted a=0.7 | auto_company | 0.137 | 0.363 | 0.494 | 0.701 | 0.252 | 0.900 | 1272 |
| chunk-tok256__rrf+rerank__auto_company | tok256 | rrf+rerank | auto_company | 0.280 | 0.540 | 0.607 | 0.647 | 0.416 | 0.940 | 1270 |

Paired bootstrap (10k resamples over questions), difference in MRR@10 and Recall@5 (B - A):

| A | B | dMRR@10 [95% CI] | dR@5 [95% CI] |
|---|---|---|---|
| chunk-tok256__bm25__auto_company | chunk-tok256__rrf__auto_company | +0.084 [+0.042, +0.126] | +0.147 [+0.077, +0.217] |
| chunk-tok256__dense__auto_company | chunk-tok256__rrf__auto_company | +0.044 [-0.001, +0.088] | +0.019 [-0.054, +0.094] |
| chunk-tok256__rrf__none | chunk-tok256__rrf__auto_company | +0.046 [+0.020, +0.075] | +0.037 [-0.003, +0.080] |
| chunk-tok256__rrf__auto_company | chunk-tok256__rrf+rerank__auto_company | +0.155 [+0.095, +0.216] | +0.200 [+0.127, +0.273] |
