import numpy as np
def scaled_exponential_linear_unit(
    vector: np.ndarray, alpha: float = 1.6732, lambda_: float = 1.0507
) -> np.ndarray:
    return lambda_ * np.where(vector > 0, vector, alpha * (np.exp(vector) - 1))
if __name__ == "__main__":
    import doctest
    doctest.testmod()