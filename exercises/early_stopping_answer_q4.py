import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def error(pred, true):
    mean_pred = np.mean(pred)
    mean_true = np.mean(true)
    return abs(mean_pred - mean_true)


def true_fun(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)

n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
error_validation = 10
i = 1
while i > 0:

    polynomial_features = PolynomialFeatures(degree=i,
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    X_test = np.linspace(0, 1, 100)
    pred = pipeline.predict(X_test[:, np.newaxis])
    true = true_fun(X_test)
    new_error_validation = error(pred, true)

    if error_validation < new_error_validation:
        ax = plt.subplot(1, 1, 1)
        print("Early stopping")
        plt.plot(X_test, pred, label="Model")
        plt.plot(X_test, true, label="True function")
        plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title("Degree {}".format(i))
        break
    error_validation = new_error_validation
    i = i+1
plt.show()