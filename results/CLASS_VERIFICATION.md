# Class verification (real held-out test images through POST /predict)

Model class mapping: `{"classes": ["Alternate", "Opposite", "Whorled", "Not_a_leaf"], "label2id": {"Alternate": 0, "Opposite": 1, "Whorled": 2, "Not_a_leaf": 3}, "id2label": {"0": "Alternate", "1": "Opposite", "2": "Whorled", "3": "Not_a_leaf"}}`

| Class | Test image | Predicted class | Confidence (calibrated) | Decision | Result |
|---|---|---|---|---|---|
| Alternate | Alternate/12.jpeg | Alternate | 0.988 | ACCEPT | PASS |
| Opposite | Opposite/12.jpeg | Opposite | 0.988 | ACCEPT | PASS |
| Whorled | Whorled/16.jpeg | Whorled | 0.980 | ACCEPT | PASS |
| Not_a_leaf | Not_a_leaf/not_a_leaf_0001.jpg | Not_a_leaf | 0.925 | NOT_A_LEAF | PASS |

One image per class is a smoke test, not an accuracy claim. Full held-out confusion matrices are printed above and stored in class_check.json (small test sets: report Wilson CIs, see results/RESULTS.md).
