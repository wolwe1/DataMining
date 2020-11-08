import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import tree


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


def plotLineGraph(title, xValues, yValues):
    plt.plot(xValues, yValues)
    setGraphDetails(title)
    plt.show()


def plotBarGraph(title, xValues, yValues):
    plt.bar(xValues, yValues)
    setGraphDetails(title)
    plt.show()


def setGraphDetails(title):
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


def plotOccurencesLine(pandasSeries, title, sort=False):
    (seriesValues, occurrencesOfValue) = getOccurenceCount(pandasSeries)

    if sort:
        seriesValues = np.array(seriesValues)
        occurrencesOfValue = np.array(occurrencesOfValue)
        inds = seriesValues.argsort()
        sortedValues = seriesValues[inds]
        sortedOccurences = occurrencesOfValue[inds]
        plotLineGraph(title, sortedValues, sortedOccurences)
    else:
        plotLineGraph(title, seriesValues, occurrencesOfValue)


def plotOccurencesBar(pandasSeries, title, sort=False):
    (values, occurrencesOfValue) = getOccurenceCount(pandasSeries)

    if sort:
        values = np.array(values)
        occurrencesOfValue = np.array(occurrencesOfValue)
        inds = values.argsort()
        sortedValues = values[inds]
        sortedOccurences = occurrencesOfValue[inds]
        plotBarGraph(sortedValues, sortedOccurences)
    else:
        plotBarGraph(values, occurrencesOfValue)


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

    i = 0

    for index, rowdata in dataColumn.iteritems():
        z = (rowdata - mean) / std
        if z > threshold:
            outliers.append(i)

        i += 1
    return outliers


def convertClassesToLabels(dataSet):
    label_encoder = LabelEncoder()

    dataSet["occupation"] = label_encoder.fit_transform(dataSet["occupation"])
    dataSet["workclass"] = label_encoder.fit_transform(dataSet["workclass"])
    dataSet["education"] = label_encoder.fit_transform(dataSet["education"])
    dataSet["marital-status"] = label_encoder.fit_transform(dataSet["marital-status"])
    dataSet["relationship"] = label_encoder.fit_transform(dataSet["relationship"])
    dataSet["race"] = label_encoder.fit_transform(dataSet["race"])
    dataSet["sex"] = label_encoder.fit_transform(dataSet["sex"])
    dataSet["native-country"] = label_encoder.fit_transform(dataSet["native-country"])
    dataSet["income"] = label_encoder.fit_transform(dataSet["income"])
    dataSet["capital-gain Group"] = label_encoder.fit_transform(dataSet["capital-gain Group"])
    dataSet["capital-loss Group"] = label_encoder.fit_transform(dataSet["capital-loss Group"])
    dataSet["fnlwgt Group"] = label_encoder.fit_transform(dataSet["fnlwgt Group"])
    dataSet["hours-per-week Group"] = label_encoder.fit_transform(dataSet["hours-per-week Group"])

    return dataSet


def createQuartiles(label, dataset):
    labels = ['1st-Quartile', '2nd-Quartile', '3rd-Quartile', '4th-Quartile']
    quartiles = pd.qcut(dataset[label], q=4, labels=labels)
    dataset.insert(5, label + ' Group', quartiles)


# endregion

# region definitions
initialColumnNames = ["age",
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
                      "income"]

columnNamesForUseInClassification = ["age",
                                     "workclass",
                                     "fnlwgt Group",
                                     "education",
                                     "marital-status",
                                     "occupation",
                                     "relationship",
                                     "race",
                                     "sex",
                                     "capital-gain Group",
                                     "capital-loss Group",
                                     "hours-per-week Group",
                                     "native-country"]
# endregion

# region Data reading
trainingSet = pd.read_csv("C:\\Users\\jarro\\Desktop\\University\\COS781\\Assignments\\Assignment2\\data\\adult.data",
                          sep=',\s', header=None, engine='python')
testingSet = pd.read_csv("C:\\Users\\jarro\\Desktop\\University\\COS781\\Assignments\\Assignment2\\data\\adult.test",
                         sep=',\s', header=None, engine='python', skiprows=1)

testingSet.columns = initialColumnNames
trainingSet.columns = initialColumnNames

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
trainingSet = trainingSet.replace('?', np.NaN)

printMissingPercentage("workclass", trainingSet)
printMissingPercentage("occupation", trainingSet)
printMissingPercentage("native-country", trainingSet)

preDropSize = trainingSet.shape[0]
trainingSet = trainingSet.dropna()
postDropSize = trainingSet.shape[0]
print("Percentage lost in drop ", (postDropSize / preDropSize * 100) - 100, " %")

# Remove garbage columns
trainingSet.drop('education-num', axis=1, inplace=True)
testingSet.drop('education-num', axis=1, inplace=True)
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
#
# numericCols = dataset.select_dtypes(include=numerics)
# print(numericCols)
# outliers = numericCols[(np.abs(stats.zscore(numericCols)) < 3).all(axis=1)]
# print("Outliers ", outliers)
#

# outlierIndexes = getOutlierIndex("age", dataset["age"])
# minIndex = min(outlierIndexes)
# maxIndex = max(outlierIndexes)
# print('Dataset size ', dataset.shape[0])
# print("Age outliers ", minIndex, " - ", maxIndex, outlierIndexes)
#
# outliers = dataset.iloc[outlierIndexes, :]
# print(outliers)
corr = trainingSet.apply(lambda x: x.factorize()[0]).corr()
plt.figure(figsize=(16, 5))
sn.heatmap(corr, annot=True, linewidths=.5)
plt.show()
# endregion

# region Decision tree

# region Prep

# Bin continuos data to categories
category = pd.cut(trainingSet.age, bins=[0, 2, 17, 65, 99], labels=['Toddler/baby', 'Child', 'Adult', 'Elderly'])
trainingSet.insert(1, 'age Group', category)

category = pd.cut(testingSet.age, bins=[0, 2, 17, 65, 99], labels=['Toddler/baby', 'Child', 'Adult', 'Elderly'])
testingSet.insert(1, 'age Group', category)

# capital gains
category = pd.cut(trainingSet["capital-gain"], bins=[-1, 1, 25000, 50000, float("inf")],
                  labels=['none', '<=25k', '<=50k', '>50k'])
trainingSet.insert(1, 'capital-gain Group', category)

category = pd.cut(testingSet["capital-gain"], bins=[-1, 1, 25000, 50000, float("inf")],
                  labels=['none', '<=25k', '<=50k', '>50k'])
testingSet.insert(1, 'capital-gain Group', category)
# capital loss
category = pd.cut(trainingSet["capital-loss"], bins=[-1, 1, 1000, 2500, float("inf")],
                  labels=['none', '<=1k', '<=2.5k', '>2.5k'])
trainingSet.insert(1, 'capital-loss Group', category)

category = pd.cut(testingSet["capital-loss"], bins=[-1, 1, 1000, 2500, float("inf")],
                  labels=['none', '<=1k', '<=2.5k', '>2.5k'])
testingSet.insert(1, 'capital-loss Group', category)

print(trainingSet[["capital-gain", "capital-gain Group"]])

createQuartiles("fnlwgt", trainingSet)
createQuartiles("fnlwgt", testingSet)

category = pd.cut(trainingSet["hours-per-week"], bins=[0, 1, 39, 41, 55, float("inf")],
                  labels=['none', 'under 40', '40', 'over 40', 'extreme'])
trainingSet.insert(1, 'hours-per-week Group', category)

category = pd.cut(testingSet["hours-per-week"], bins=[0, 1, 39, 41, 55, float("inf")],
                  labels=['none', 'under 40', '40', 'over 40', 'extreme'])
testingSet.insert(1, 'hours-per-week Group', category)


trainingSet = convertClassesToLabels(trainingSet)
testingSet = convertClassesToLabels(testingSet)

x_train = trainingSet[columnNamesForUseInClassification]  # Features
x_test = testingSet[columnNamesForUseInClassification]

y_train = trainingSet["income"]
y_test = testingSet["income"]

# endregion
clf = DecisionTreeClassifier()

clf = clf.fit(x_train, y_train)

trainingPredictions = clf.predict(x_train)
print("Training Accuracy:", metrics.accuracy_score(y_train, trainingPredictions) * 100, " %")

testingPredictions = clf.predict(x_test)
print("Testing Accuracy:", metrics.accuracy_score(y_test, testingPredictions) * 100, " %")

fig = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(clf,
                   feature_names=columnNamesForUseInClassification,
                   class_names=[">50K", "<=50K"],
                   filled=True)
fig.savefig("decision_tree.png")
# endregion
