This project trains a **binary classifier** on tabular features using a
**Keras MLP (Dense network)** with:
- **Train/val/test splits** inside a 5-iteration loop
- **SMOTE** oversampling on the training split to handle class imbalance
- **Class weights** for the loss
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Evaluation**: ROC curves per run, confusion matrix–derived metrics,
  and loss/accuracy plots

## Data format

The code expects a CSV with at least these columns:

- Features (example from the script):
  - `Cor`, `TTP`, `MTT_SVD`, `PH`, `AUC`, `Max_Gr`, `CBF`, `CBV`
- Target:
  - `Target` (binary: 0/1)

