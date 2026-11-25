# Model comparison

- Subjects/groups: nthu001, nthu002, uta
- Rows: 19278 (train: 15377, test: 3901)
- Split: UTA-RLDD official folders (test = test/), NTHU GroupShuffleSplit by subject (test_size=0.5)
- NTHU test subjects: nthu002
- Class weight: balanced (random_forest, svm_rbf)
- Thresholds (baseline): PERCLOS > 0.2, |yaw_mean| > 20.0 deg

## Summary

| Model | Accuracy | Macro-F1 | Drowsy precision | Drowsy recall | Drowsy F1 |
|---|---|---|---|---|---|
| threshold | 0.326 | 0.295 | 0.823 | 0.395 | 0.534 |
| random_forest | 0.774 | 0.510 | 0.734 | 0.908 | 0.812 |
| svm_rbf | 0.762 | 0.498 | 0.713 | 0.931 | 0.808 |

## threshold

```
              precision    recall  f1-score   support

       alert       0.62      0.25      0.35      1806
      drowsy       0.82      0.39      0.53      2095
  distracted       0.00      0.00      0.00         0

    accuracy                           0.33      3901
   macro avg       0.48      0.21      0.30      3901
weighted avg       0.73      0.33      0.45      3901

Confusion matrix (rows=true, cols=pred):
[[ 444  178 1184]
 [ 272  827  996]
 [   0    0    0]]
```

## random_forest

```
              precision    recall  f1-score   support

       alert       0.85      0.62      0.72      1806
      drowsy       0.73      0.91      0.81      2095
  distracted       0.00      0.00      0.00         0

    accuracy                           0.77      3901
   macro avg       0.53      0.51      0.51      3901
weighted avg       0.79      0.77      0.77      3901

Confusion matrix (rows=true, cols=pred):
[[1117  689    0]
 [ 193 1902    0]
 [   0    0    0]]
```

## svm_rbf

```
              precision    recall  f1-score   support

       alert       0.88      0.57      0.69      1806
      drowsy       0.71      0.93      0.81      2095
  distracted       0.00      0.00      0.00         0

    accuracy                           0.76      3901
   macro avg       0.53      0.50      0.50      3901
weighted avg       0.79      0.76      0.75      3901

Confusion matrix (rows=true, cols=pred):
[[1022  784    0]
 [ 145 1950    0]
 [   0    0    0]]
```
