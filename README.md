# FIW-SVD

Implementation of FIW-SVD (Feature Importance Weighted SVD) for task-aware missing value imputation.

This repository includes:

* KNN imputation
* MICE imputation
* FIW-SVD
* Multi-modal imputation framework (MM)
* Classification and regression experiments
* Optional SMOTE-based classification experiments

## Requirements

```bash
pip install scikit-learn imbalanced-learn xgboost pandas numpy
```

## Dataset

Place `city_day.csv` in the project directory.

## Run

### Default

```bash
python main_10.py
```

### Multiple kappa values

```bash
python main.py --kappa_list 0 1 3 5 10 30 50
```

### With SMOTE

```bash
python main_10.py --run_smote
```

## Outputs

The script saves:

* `classification_results.csv`
* `regression_results.csv`

Results contain mean and standard deviation over 5 random 70/10/20 train-validation-test splits.

## Main Components

* `KNN`
* `MICE`
* `SVD(FIW)`
* `MM(FIW)` = multi-modal combination of KNN + MICE + FIW-SVD

## Citation

If you use this code, please cite our paper.
