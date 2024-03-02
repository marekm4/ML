from sklearn.datasets import load_iris

iris = load_iris()

versicolor_sepal_length = [
    length for i, length in enumerate(iris.data[:, 0]) if iris.target[i] == 1
]
virginica_sepal_length = [
    length for i, length in enumerate(iris.data[:, 0]) if iris.target[i] == 2
]
print(versicolor_sepal_length)
print(virginica_sepal_length)

from scipy.stats import ttest_ind

result = ttest_ind(versicolor_sepal_length, virginica_sepal_length)
print(result.pvalue)
