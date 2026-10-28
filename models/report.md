# Model comparison

- Subjects: dry-a, dry-b, dry-c, real_1, real_2, real_3, real_4, real_5
- Rows: 10036
- Evaluation: GroupKFold by subject (5 folds), pooled out-of-fold predictions
- Thresholds (baseline): PERCLOS > 0.2, |yaw_mean| > 20.0 deg

## Summary

| Model | Accuracy | Macro-F1 | Drowsy precision | Drowsy recall | Drowsy F1 |
|---|---|---|---|---|---|
| threshold | 0.975 | 0.964 | 0.995 | 0.999 | 0.997 |
| random_forest | 0.999 | 0.998 | 1.000 | 1.000 | 1.000 |
| svm_rbf | 0.999 | 0.998 | 1.000 | 1.000 | 1.000 |

## threshold

```
              precision    recall  f1-score   support

       alert       0.96      1.00      0.98      6012
      drowsy       1.00      1.00      1.00      2512
  distracted       1.00      0.84      0.91      1512

    accuracy                           0.98     10036
   macro avg       0.99      0.95      0.96     10036
weighted avg       0.98      0.98      0.97     10036

Confusion matrix (rows=true, cols=pred):
[[6012    0    0]
 [   3 2509    0]
 [ 231   12 1269]]
```

## random_forest

```
              precision    recall  f1-score   support

       alert       1.00      1.00      1.00      6012
      drowsy       1.00      1.00      1.00      2512
  distracted       0.99      1.00      1.00      1512

    accuracy                           1.00     10036
   macro avg       1.00      1.00      1.00     10036
weighted avg       1.00      1.00      1.00     10036

Confusion matrix (rows=true, cols=pred):
[[6003    0    9]
 [   0 2512    0]
 [   2    0 1510]]
```

## svm_rbf

```
              precision    recall  f1-score   support

       alert       1.00      1.00      1.00      6012
      drowsy       1.00      1.00      1.00      2512
  distracted       1.00      0.99      1.00      1512

    accuracy                           1.00     10036
   macro avg       1.00      1.00      1.00     10036
weighted avg       1.00      1.00      1.00     10036

Confusion matrix (rows=true, cols=pred):
[[6012    0    0]
 [   0 2512    0]
 [  15    0 1497]]
```
