import numpy as np


def create_leaf(label):
    return {'isLeaf': True, 'label': label}


def create_node(left, right, attr, value):
    return {
        'isLeaf': False,
        'attr': attr,
        'value': value,
        'left': left,
        'right': right
    }


def calculate_entropy(dataSet: np.array) -> float:
    labelsArray = dataSet[:, -1]
    _, labelsCount = np.unique(labelsArray, return_counts=True)
    size = len(labelsArray)
    entropy = 0
    for count in labelsCount:
        freq = count / size
        entropy -= freq * np.log2(freq)
    return entropy


def calculate_info_gain(leftDataSet, rightDataSet, dataSetEntropy):
    entropyLeft = calculate_entropy(leftDataSet)
    entropyRight = calculate_entropy(rightDataSet)

    lenLeft = len(leftDataSet)
    lenRight = len(rightDataSet)
    numRows = lenLeft + lenRight

    remainder = (lenLeft / numRows) * entropyLeft + (lenRight /
                                                     numRows) * entropyRight
    return dataSetEntropy - remainder


def find_split(dataSet: np.array):
    dataSetEntropy = calculate_entropy(dataSet)
    # '-1' to exclude labels column
    numFeatureColumns = len(dataSet[0]) - 1
    numRows = len(dataSet)

    maxInfoGain = bestAttrColumn = boundaryValue = 0
    optimalLeftDataSet = optimalRightDataSet = None

    for c in range(numFeatureColumns):
        # Sort data set by attribute
        sortedDataSet = dataSet[np.argsort(dataSet[:, c], kind='mergesort')]

        uniqueValues, uniqueIndices = np.unique(sortedDataSet[:, c],
                                                return_index=True)

        for splitIdx in uniqueIndices:
            [leftDataSet, rightDataSet] = np.split(sortedDataSet, [splitIdx])
            currInfoGain = calculate_info_gain(leftDataSet, rightDataSet,
                                               dataSetEntropy)

            if currInfoGain > maxInfoGain:
                maxInfoGain = currInfoGain
                bestAttrColumn, boundaryValue = c, sortedDataSet[splitIdx][c]
                optimalLeftDataSet, optimalRightDataSet = leftDataSet, rightDataSet

    return bestAttrColumn, boundaryValue, optimalLeftDataSet, optimalRightDataSet


def decision_tree_learning(trainingDataSet, depth=0):
    labelsArray = trainingDataSet[:, -1]
    uniqueLabels = np.unique(labelsArray)
    if len(uniqueLabels) == 1:
        return create_leaf(uniqueLabels[0]), depth
    else:
        bestAttrColumn, boundaryValue, optimalLeftDataSet, optimalRightDataSet = find_split(
            trainingDataSet)
        lBranch, lDepth = decision_tree_learning(optimalLeftDataSet, depth + 1)
        rBranch, rDepth = decision_tree_learning(optimalRightDataSet, depth + 1)
        node = create_node(lBranch, rBranch, bestAttrColumn, boundaryValue)
        return node, max(lDepth, rDepth)
