# Anomaly Evaluation

Created by School of AI and School of QC.

Every detector emits a raw anomaly score. For the QAE this is reconstruction infidelity; for PCA
it is mean squared reconstruction error; for Isolation Forest it is negated normality score.
Each score's minimum and maximum are learned on validation data
and used to map scores to `[0, 1]`. A sweep over 201 candidate thresholds minimizes

$$C = c_{FN}FN + c_{FP}FP.$$

The default assigns a missed anomaly five times the cost of a false alarm. Ties prefer higher
validation recall and then the higher threshold. The chosen threshold and score range are frozen
before the test set is evaluated.

The report includes precision, recall, specificity, F1, F2, balanced accuracy, Matthews
correlation, ROC AUC, average precision, confusion counts, lift at ten percent, timing, and the
configured business cost. Average precision is especially useful for the imbalanced test set;
the confusion matrix makes the selected operating point explicit.

The predictions artifact preserves both raw and normalized scores. The anomaly-family artifact
reports counts, mean scores, flagged rates, and QAE mean fidelity separately for normal, sensor
spike, process drift, correlation break, and stuck-sensor inputs. Small family counts should not
be treated as uncertainty estimates.

These are synthetic benchmark measurements, not production guarantees. Recalibrate against
representative data and decide costs with affected domain experts before any real deployment.
