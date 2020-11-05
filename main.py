import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import miceforest as mf
import scipy.stats as stats


# region Functions
def getrange(numbers):
    return max(numbers) - min(numbers)


def getDistinct(list):
    return set(list)


def getOccurenceCount(arrList):
    occurrences = {}
    for row in arrList:
        if str(row) not in occurrences:
            occurrences.setdefault(str(row), 1)
        else:
            occurrences[str(row)] += 1

    assert sum(occurrences.values()) == len(arrList)
    keys = list(occurrences.keys())
    values = list(occurrences.values())
    return (keys, values)


def plotScatter(pandasSeries, title, sort=False):
    (values, occurrencesOfValue) = getOccurenceCount(pandasSeries)
    area = np.pi * 3
    colors = (0, 0, 0)
    if sort:
        values = np.array(values)
        occurrencesOfValue = np.array(occurrencesOfValue)
        inds = values.argsort()
        sortedValues = values[inds]
        sortedOccurences = occurrencesOfValue[inds]
        plt.scatter(sortedValues, sortedOccurences, s=area, c=colors, alpha=0.5)

    else:
        plt.scatter(values, occurrencesOfValue, s=area, c=colors, alpha=0.5)
    # plt.autoscale(enable=True, axis='x', tight=None)
    plt.title(title)
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * len(values) + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.show()


def plotOccurencesLine(pandasSeries, title, sort=False):
    (values, occurrencesOfValue) = getOccurenceCount(pandasSeries)

    if sort:
        values = np.array(values)
        occurrencesOfValue = np.array(occurrencesOfValue)
        inds = values.argsort()
        sortedValues = values[inds]
        sortedOccurences = occurrencesOfValue[inds]
        plt.plot(sortedValues, sortedOccurences)
    else:
        plt.plot(values, occurrencesOfValue)
    # plt.autoscale(enable=True, axis='x', tight=None)
    plt.title(title)
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * len(values) + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.show()


def plotOccurencesBar(pandasSeries, title, sort=False):
    (values, occurrencesOfValue) = getOccurenceCount(pandasSeries)

    if sort:
        values = np.array(values)
        occurrencesOfValue = np.array(occurrencesOfValue)
        inds = values.argsort()
        sortedValues = values[inds]
        sortedOccurences = occurrencesOfValue[inds]
        plt.bar(sortedValues, sortedOccurences)
    else:
        plt.bar(values, occurrencesOfValue)

    # plt.autoscale(enable=True, axis='x', tight=None)
    plt.title(title)
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    tl = plt.gca().get_xticklabels()
    maxsize = max([t.get_window_extent().width for t in tl])
    m = 0.2  # inch margin
    s = maxsize / plt.gcf().dpi * len(values) + 2 * m
    margin = m / plt.gcf().get_size_inches()[0]

    plt.gcf().subplots_adjust(left=margin, right=1. - margin)
    plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
    plt.show()


def printMissingPercentage(field, dataset):
    missing = dataset[field].isnull().sum()
    size = dataset[field].size
    print(missing, size)
    print("Missing values for", field, missing, " , ", (missing / size) * 100, "%")


def getOutlierIndex(columnName, dataColumn):
    mean = np.mean(dataColumn)
    std = np.std(dataColumn)

    threshold = 3
    outliers = []

    print('mean of ', columnName, ' is', mean)
    print('std. deviation is', std)

    for index, rowdata in dataColumn.iteritems():
        z = (rowdata - mean) / std
        if z > threshold:
            outliers.append(index)

    return outliers


# endregion

# region Data reading
dataset = pd.read_csv("C:\\Users\\jarro\\Desktop\\University\\COS781\\Assignments\\Assignment2\\data\\adult.data",
                      sep=',\s', header=None, engine='python');
dataset.columns = ["age",
                   "workclass",
                   "fnlwgt",
                   "education",
                   "education-num",
                   "marital-status",
                   "occupation",
                   "relationship",
                   "race",
                   "sex",
                   "capital-gain",
                   "capital-loss",
                   "hours-per-week",
                   "native-country",
                   "income"
                   ]

# endregion

# region Plotting

# plotOccurencesLine(dataset["age"], "age")
# plotOccurencesBar(dataset["workclass"], "workclass")
# plotOccurences(dataset["fnlwgt"],"fnlwgt")
# plotOccurencesBar(dataset["education"], "education")

# plotOccurencesBar(dataset["marital-status"], "marital-status")
# plotOccurencesBar(dataset["occupation"], "occupation")
# plotOccurencesBar(dataset["relationship"], "relationship")
# plotOccurencesBar(dataset["race"], "race")
# plotOccurencesBar(dataset["sex"], "sex")
# plotOccurencesLine(dataset["capital-gain"],"capital-gain")
# plotOccurencesLine(dataset["capital-loss"],"capital-loss")
# plotOccurencesBar(dataset["hours-per-week"],"hours-per-week",sort=True)
# plotOccurencesBar(dataset["native-country"],"native-country")

# plotScatter(dataset["hours-per-week"],"hours-per-week",True)

# endregion

# region Counts
# print(dataset.age.value_counts())
# print(dataset.workclass.value_counts())
# print(dataset.fnlwgt.value_counts())
# print(dataset.education.value_counts())
# print(dataset["marital-status"].value_counts())
# print(dataset.occupation.value_counts())
# print(dataset.relationship.value_counts())
# print(dataset.race.value_counts())
# print(dataset.sex.value_counts())
# print(dataset["capital-gain"].value_counts())
# print(dataset["capital-loss"].value_counts())
# print(dataset["hours-per-week"].value_counts())
# print(dataset["native-country"].value_counts())

# endregion

# region DataPreperation
dataset = dataset.replace('?', np.NaN)

printMissingPercentage("workclass", dataset)
printMissingPercentage("occupation", dataset)
printMissingPercentage("native-country", dataset)

preDropSize = dataset.shape[0]
dataset = dataset.dropna()
postDropSize = dataset.shape[0]
print("Percentage lost in drop ", (postDropSize / preDropSize * 100) - 100, " %")

# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#
# numericCols = dataset.select_dtypes(include=numerics)
# print(numericCols)
# outliers = numericCols[(np.abs(stats.zscore(numericCols)) < 3).all(axis=1)]
# print("Outliers ", outliers)
#

#TODO: Track down whats causing the size to go down
outlierIndexes = getOutlierIndex("age", dataset["age"])
minIndex = min(outlierIndexes)
maxIndex = max(outlierIndexes)
print('Dataset size ', dataset.shape[0])
print("Age outliers ", minIndex, " - ", maxIndex, outlierIndexes)

outliers = dataset.iloc[outlierIndexes, :]
print(outliers)
