import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import datasets

pd.set_option('display.max_columns', None)
sns.set()


# How is this extracted into a shared fn?
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y


iris = datasets.load_iris()

iris_as_df = sklearn_to_df(iris)
iris_as_df.loc[:, 'target'] = [iris.target_names[i] for i in iris_as_df['target']]

versicolor_petal_length = iris_as_df.loc[iris_as_df['target'] == 'versicolor', "petal length (cm)"]
versicolor_petal_width = iris_as_df.loc[iris_as_df['target'] == 'versicolor', "petal width (cm)"]
setosa_petal_length = iris_as_df.loc[iris_as_df['target'] == 'setosa', "petal length (cm)"]
virginica_petal_length = iris_as_df.loc[iris_as_df['target'] == 'virginica', "petal length (cm)"]

# Specify array of percentiles: percentiles
percentiles = np.array([2.5, 25, 50, 75, 97.5])

# Compute percentiles: ptiles_vers
# Easy to see percentiles in an ECDF plot
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)

# Print the result
print(ptiles_vers)

# Theory: [3.3    4.     4.35   4.6    4.9775]
# 75th percentile > 75% of the versicolor have 4.6cm length or less

x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
_ = plt.figure(1)
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel("petal length (cm)")
_ = plt.ylabel("ECDF")
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red', linestyle='none')

# The box shows the inter-quartile range or IQR
# The lines the extent of data and individual points outside are outliers
_ = plt.figure(2)
_ = sns.boxplot(x='target', y='petal length (cm)', data=iris_as_df)
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')

_ = plt.figure(3)
_ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('petal width (cm)')

# Compute the covariance matrix: covariance_matrix
# Entries [0,1] and [1,0] are the covariances
# Entry [0,0] is the variance of the data in x
# Entry [1,1] is the variance of the data in y
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0, 1]

# Print the length/width covariance
print(petal_cov)

plt.show()
