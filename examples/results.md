## english: 120 examples, labels {'new_card': 30, 'card_security': 30, 'transfers': 30, 'top_up': 30}; 20 to build from, 100 held out per sample, pass rate 0.85

Score on the 20 (leave one out) / score on the held-out part, in per cent. Sample 0 is the repository's 20; 1 to 5 are random.

| model | s0 | s1 | s2 | s3 | s4 | s5 | mean 1-5 | passes 1-5 | of them held < 80 |
|---|---|---|---|---|---|---|---|---|---|
| all-minilm-l6-v2 | 95 / 83 | 85 / 87 | 90 / 81 | 85 / 83 | 95 / 80 | 80 / 86 | 87 / 83 | 4/5 | 0 |
| bge-small-en-v1.5 | 100 / 86 | 90 / 91 | 90 / 88 | 85 / 87 | 95 / 82 | 90 / 88 | 90 / 87 | 5/5 | 0 |
| embeddinggemma-300m | 100 / 95 | 90 / 96 | 95 / 93 | 90 / 90 | 95 / 87 | 95 / 93 | 93 / 92 | 5/5 | 0 |
| multilingual-e5-small | 90 / 83 | 65 / 82 | 95 / 79 | 80 / 79 | 85 / 75 | 80 / 83 | 81 / 80 | 2/5 | 2 |
| paraphrase-multilingual-minilm-l12-v2 | 90 / 77 | 85 / 81 | 90 / 82 | 90 / 80 | 90 / 80 | 75 / 84 | 86 / 81 | 4/5 | 0 |

Model taken by `selection.select`, and its score on the held-out part:

| sample | license: permissive | license: any |
|---|---|---|
| s0 | all-minilm-l6-v2: 95 / 83 | all-minilm-l6-v2: 95 / 83 |
| s1 | all-minilm-l6-v2: 85 / 87 | all-minilm-l6-v2: 85 / 87 |
| s2 | all-minilm-l6-v2: 90 / 81 | all-minilm-l6-v2: 90 / 81 |
| s3 | all-minilm-l6-v2: 85 / 83 | all-minilm-l6-v2: 85 / 83 |
| s4 | all-minilm-l6-v2: 95 / 80 | all-minilm-l6-v2: 95 / 80 |
| s5 | bge-small-en-v1.5: 90 / 88 | bge-small-en-v1.5: 90 / 88 |

license permissive: no model passes on the 20 in 0 of 5 random samples []; the chosen model has a held-out score below 80% in 0 of them []
license any: no model passes on the 20 in 0 of 5 random samples []; the chosen model has a held-out score below 80% in 0 of them []

| model | 20 examples, wall | model load | an example (first job) | an example (all jobs) | wall per example, held-out jobs |
|---|---|---|---|---|---|
| all-minilm-l6-v2 | 2.8 s | 1.7 s | 12 ms | 12 ms | 0.12 s |
| bge-small-en-v1.5 | 3.9 s | 2.3 s | 44 ms | 40 ms | 0.16 s |
| embeddinggemma-300m | 32.7 s | 6.8 s | 1219 ms | 1317 ms | 1.60 s |
| multilingual-e5-small | 6.9 s | 5.5 s | 37 ms | 36 ms | 0.27 s |
| paraphrase-multilingual-minilm-l12-v2 | 6.3 s | 4.9 s | 36 ms | 34 ms | 0.27 s |

## russian: 120 examples, labels {'будильник': 30, 'календарь': 30, 'почта': 30, 'соцсети': 30}; 20 to build from, 100 held out per sample, pass rate 0.85

Score on the 20 (leave one out) / score on the held-out part, in per cent. Sample 0 is the repository's 20; 1 to 5 are random.

| model | s0 | s1 | s2 | s3 | s4 | s5 | mean 1-5 | passes 1-5 | of them held < 80 |
|---|---|---|---|---|---|---|---|---|---|
| all-minilm-l6-v2 | 40 / 66 | 65 / 59 | 55 / 56 | 65 / 58 | 50 / 65 | 50 / 61 | 57 / 60 | 0/5 | 0 |
| bge-small-en-v1.5 | 55 / 66 | 70 / 56 | 60 / 65 | 55 / 63 | 35 / 56 | 65 / 57 | 57 / 59 | 0/5 | 0 |
| embeddinggemma-300m | 85 / 93 | 95 / 100 | 100 / 99 | 85 / 97 | 90 / 97 | 90 / 99 | 92 / 98 | 5/5 | 0 |
| multilingual-e5-small | 75 / 82 | 65 / 83 | 95 / 84 | 80 / 79 | 75 / 85 | 80 / 78 | 79 / 82 | 1/5 | 0 |
| paraphrase-multilingual-minilm-l12-v2 | 90 / 89 | 70 / 88 | 90 / 83 | 80 / 85 | 85 / 81 | 80 / 84 | 81 / 84 | 2/5 | 0 |

Model taken by `selection.select`, and its score on the held-out part:

| sample | license: permissive | license: any |
|---|---|---|
| s0 | paraphrase-multilingual-minilm-l12-v2: 90 / 89 | paraphrase-multilingual-minilm-l12-v2: 90 / 89 |
| s1 | none | embeddinggemma-300m: 95 / 100 |
| s2 | multilingual-e5-small: 95 / 84 | multilingual-e5-small: 95 / 84 |
| s3 | none | embeddinggemma-300m: 85 / 97 |
| s4 | paraphrase-multilingual-minilm-l12-v2: 85 / 81 | paraphrase-multilingual-minilm-l12-v2: 85 / 81 |
| s5 | none | embeddinggemma-300m: 90 / 99 |

license permissive: no model passes on the 20 in 3 of 5 random samples [1, 3, 5]; the chosen model has a held-out score below 80% in 0 of them []
license any: no model passes on the 20 in 0 of 5 random samples []; the chosen model has a held-out score below 80% in 0 of them []

| model | 20 examples, wall | model load | an example (first job) | an example (all jobs) | wall per example, held-out jobs |
|---|---|---|---|---|---|
| all-minilm-l6-v2 | 2.6 s | 1.7 s | 17 ms | 17 ms | 0.11 s |
| bge-small-en-v1.5 | 3.2 s | 2.0 s | 28 ms | 38 ms | 0.15 s |
| embeddinggemma-300m | 33.3 s | 6.5 s | 1291 ms | 1282 ms | 1.51 s |
| multilingual-e5-small | 6.4 s | 5.0 s | 32 ms | 32 ms | 0.25 s |
| paraphrase-multilingual-minilm-l12-v2 | 6.1 s | 4.7 s | 31 ms | 29 ms | 0.26 s |

