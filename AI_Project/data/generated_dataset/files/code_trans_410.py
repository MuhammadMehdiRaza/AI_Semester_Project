"""
this is code for forecasting
but I modified it and used it for safety checker of data
for ex: you have an online shop and for some reason some data are
missing (the amount of data that u expected are not supposed to be)
        then we can use it
*ps : 1. ofc we can use normal statistic method but in this case
         the data is quite absurd and only a little^^
      2. ofc u can use this and modified it for forecasting purpose
         for the next 3 months sales or something,
         u can just adjust it for ur own purpose
"""

from warnings import simplefilter

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fn_3(
    val_31: list, val_35: list, val_33: list, val_23: list, val_25: list
) -> float:
    """
    First method: linear regression
    input : training data (date, val_29, total_event) in list of float
    output : list of total user prediction in float
    >>> n = fn_3([2,3,4,5], [5,3,4,6], [3,1,2,4], [2,1], [2,2])
    >>> bool(abs(n - 5.0) < 1e-6)  # Checking precision because of floating point errors
    True
    """
    val_36 = np.array([[1, val_6, val_33[val_4]] for val_4, val_6 in enumerate(val_31)])
    val_39 = np.array(val_35)
    val_2 = np.dot(np.dot(np.linalg.inv(np.dot(val_36.transpose(), val_36)), val_36.transpose()), val_39)
    return abs(val_2[0] + val_23[0] * val_2[1] + val_25[0] + val_2[2])


def fn_4(val_34: list, val_32: list, val_24: list) -> float:
    """
    second method: Sarimax
    sarimax is a statistic method which using previous input
    and learn its pattern to predict future data
    input : training data (val_29, with exog data = total_event) in list of float
    output : list of total user prediction in float
    >>> fn_4([4,2,6,8], [3,1,2,4], [2])
    6.6666671111109626
    """
    # Suppress the User Warning raised by SARIMAX due to insufficient observations
    simplefilter("ignore", UserWarning)
    val_14 = (1, 2, 1)
    val_21 = (1, 1, 1, 7)
    val_9 = SARIMAX(
        val_34, exog=val_32, val_14=val_14, val_21=val_21
    )
    val_10 = val_9.fit(disp=False, maxiter=600, method="nm")
    val_19 = val_10.predict(1, len(val_24), exog=[val_24])
    return float(val_19[0])


def fn_5(val_38: list, val_37: list, val_34: list) -> float:
    """
    Third method: Support vector val_17
    svr is quite the same with svm(support vector machine)
    it uses the same principles as the SVM for classification,
    with only a few minor differences and the only different is that
    it suits better for regression purpose
    input : training data (date, val_29, total_event) in list of float
    where val_36 = list of set (date and total event)
    output : list of total user prediction in float
    >>> fn_5([[5,2],[1,5],[6,2]], [[3,2]], [2,1,4])
    1.634932078116079
    """
    val_17 = SVR(kernel="rbf", C=1, gamma=0.1, epsilon=0.1)
    val_17.fit(val_38, val_34)
    val_40 = val_17.predict(val_37)
    return float(val_40[0])


def fn_2(val_34: list) -> float:
    """
    Optional method: interquatile range
    input : list of total user in float
    output : low limit of input in float
    this method can be used to check whether some data is outlier or not
    >>> fn_2([1,2,3,4,5,6,7,8,9,10])
    2.8
    """
    val_34.sort()
    val_15 = np.percentile(val_34, 25)
    val_16 = np.percentile(val_34, 75)
    val_5 = val_16 - val_15
    val_8 = val_15 - (val_5 * 0.1)
    return float(val_8)


def fn_1(val_7: list, val_1: float) -> bool:
    """
    Used to review all the votes (list val_19 prediction)
    and compare it to the actual val_19.
    input : list of predictions
    output : print whether it's val_20 or not
    >>> fn_1([2, 3, 4], 5.0)
    False
    """
    val_20 = 0
    val_12 = 0

    if not isinstance(val_1, float):
        raise TypeError("Actual val_19 should be float. Value passed is a list")

    for val_4 in val_7:
        if val_4 > val_1:
            val_20 = val_12 + 1
        elif abs(abs(val_4) - abs(val_1)) <= 0.1:
            val_20 += 1
        else:
            val_12 += 1
    return val_20 > val_12


if __name__ == "__main__":
    """
    data column = total user in a day, how much online event held in one day,
    what day is that(sunday-saturday)
    """
    val_3 = pd.read_csv("ex_data.csv")

    # start normalization
    val_11 = Normalizer().fit_transform(val_3.values)
    # split data
    val_27 = val_11[:, 2].tolist()
    val_29 = val_11[:, 0].tolist()
    val_28 = val_11[:, 1].tolist()

    # for svr (input variable = total date and total match)
    val_36 = val_11[:, [1, 2]].tolist()
    val_38 = val_36[: len(val_36) - 1]
    val_37 = val_36[len(val_36) - 1 :]

    # for linear regression & sarimax
    val_30 = val_27[: len(val_27) - 1]
    val_34 = val_29[: len(val_29) - 1]
    val_32 = val_28[: len(val_28) - 1]

    val_22 = val_27[len(val_27) - 1 :]
    val_26 = val_29[len(val_29) - 1 :]
    val_24 = val_28[len(val_28) - 1 :]

    # voting system with forecasting
    val_18 = [
        fn_3(
            val_30, val_34, val_32, val_22, val_24
        ),
        fn_4(val_34, val_32, val_24),
        fn_5(val_38, val_37, val_34),
    ]

    # check the safety of today's data
    val_13 = "" if fn_1(val_18, val_26[0]) else "not "
    print(f"Today's data is {val_13}val_20.")
