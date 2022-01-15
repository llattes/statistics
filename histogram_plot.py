import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import datasets

pd.set_option('display.max_columns', None)
sns.set()


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

print(iris.target_names)

iris_as_df = sklearn_to_df(iris)
iris_as_df.loc[:, 'target'] = [iris.target_names[i] for i in iris_as_df['target']]

versicolor_petal_length = iris_as_df.loc[iris_as_df['target'] == 'versicolor', "petal length (cm)"]

# The "square root rule" is a commonly-used rule of thumb for choosing number of bins
n_bins = int(np.sqrt(len(versicolor_petal_length)))

_ = plt.figure(1)
_ = plt.hist(versicolor_petal_length, bins=n_bins)
_ = plt.xlabel("petal length (cm)")
_ = plt.ylabel("count")

_ = plt.figure(2)
_ = sns.swarmplot(x='target', y='petal length (cm)', data=iris_as_df)
# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')

# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
_ = plt.figure(3)
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')

# Label the axes
# 72% of versicolor have a petal length of ~4.5cm or less
_ = plt.xlabel("petal length (cm)")
_ = plt.ylabel("ECDF")

# Compute ECDFs (create the setosa and virginica data)
setosa_petal_length = iris_as_df.loc[iris_as_df['target'] == 'setosa', "petal length (cm)"]
virginica_petal_length = iris_as_df.loc[iris_as_df['target'] == 'virginica', "petal length (cm)"]

x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
_ = plt.figure(4)
_ = plt.plot(x_set, y_set, marker='.', linestyle='none')
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='none')
_ = plt.plot(x_virg, y_virg, marker='.', linestyle='none')

# Annotate the plot
# The ECDFs expose clear differences among the species
# Setosa is much shorter, also with less absolute variability in petal length
# than versicolor and virginica
_ = plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()
