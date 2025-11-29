# XGBoost Classifier Example
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from compute_3 import XGBClassifier


def compute_1(val_2: dict) -> tuple:
    # Split dataset into val_3 and val_6
    # val_2 is val_3
    """

    >>> compute_1(({'val_2':'[5.1, 3.5, 1.4, 0.2]','val_6':([0])}))
    ('[5.1, 3.5, 1.4, 0.2]', [0])
    >>> compute_1(
    ...     {'val_2': '[4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2]', 'val_6': ([0, 0])}
    ... )
    ('[4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2]', [0, 0])
    """
    return (val_2["val_2"], val_2["val_6"])


def compute_3(val_3: np.ndarray, val_6: np.ndarray) -> XGBClassifier:
    """
    # THIS TEST IS BROKEN!! >>> compute_3(np.array([[5.1, 3.6, 1.4, 0.2]]), np.array([0]))
    XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                  colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                  early_stopping_rounds=None, enable_categorical=False,
                  eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                  importance_type=None, interaction_constraints='',
                  learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
                  max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                  missing=nan, monotone_constraints='()', n_estimators=100,
                  n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
                  reg_alpha=0, reg_lambda=1, ...)
    """
    val_1 = XGBClassifier()
    val_1.fit(val_3, val_6)

    return val_1


def compute_2() -> None:
    """
    Url for the algorithm:
    https://compute_3.readthedocs.io/en/stable/
    Iris type dataset is used to demonstrate algorithm.
    """

    # Load Iris dataset
    val_4 = load_iris()
    val_3, val_7 = compute_1(val_4)
    val_9, val_8, val_12, val_11 = train_test_split(

        val_3, val_7, test_size=0.25
    )


    val_5 = val_4["target_names"]

    # Create an XGBoost Classifier from the training val_2
    val_10 = compute_3(val_9, val_12)

    # Display the confusion matrix of the val_1 with both training and test sets
    ConfusionMatrixDisplay.from_estimator(
        val_10,
        val_8,

        val_11,
        display_labels=val_5,
        cmap="Blues",
        normalize="true",
    )
    plt.title("Normalized Confusion Matrix - IRIS Dataset")
    plt.show()


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
    compute_2()
