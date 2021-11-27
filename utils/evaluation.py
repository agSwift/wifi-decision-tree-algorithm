import numpy as np

from utils.decisionTree import decision_tree_learning


def shuffle_split_data_into_folds(dataset, num_folds=10, seed=42):
    """
    Shuffles dataset and then split into folds
    """
    shuffled_dataset = np.random.RandomState(seed=seed).permutation(dataset)
    folds = np.split(shuffled_dataset, num_folds)
    return folds


def _infer(row, tree):
    """
    Given a tree, predict a label given the attributes in the row
    """
    curr = tree
    while (not curr['isLeaf']):
        if row[curr['attr']] < curr['value']:
            curr = curr['left']
        else:
            curr = curr['right']
    return curr['label']


def evaluate(test_dataset: np.array, trained_tree):
    """
    Given a test dataset and the trained decision tree, evaluate the accuracy
    """
    predictions = np.apply_along_axis(_infer,
                                      axis=1,
                                      arr=test_dataset,
                                      tree=trained_tree)
    return predictions, accuracy(test_dataset[:, -1], predictions)


def perform_cross_validation(dataset: np.array, n_folds: int):
    """
    Perform cross validation on a dataset by splitting it into 'n_folds' number of folds
    """
    folds = shuffle_split_data_into_folds(dataset, n_folds)
    sum_accuracy = 0
    unique_labels = np.unique(dataset[:, -1])
    confusion_matrix_sum = np.zeros((len(unique_labels), len(unique_labels)))
    for i in range(n_folds):
        test_dataset = folds[i]
        training_dataset = np.concatenate(folds[:i] + folds[i + 1:])
        tree, _ = decision_tree_learning(training_dataset)
        predictions, accuracy = evaluate(test_dataset, tree)
        conf_matrix = confusion_matrix(test_dataset[:, -1], predictions)
        confusion_matrix_sum += conf_matrix
        sum_accuracy += accuracy
    return confusion_matrix_sum / n_folds, sum_accuracy / n_folds


def perform_nested_cross_validation(dataset: np.array, n_folds: int):
    """
    Performs nested cross validatin based on a (n_folds-2)/1/1 train/val/test split
    """

    folds = shuffle_split_data_into_folds(dataset, n_folds)
    sum_accuracy = 0
    unique_labels = np.unique(dataset[:, -1])
    confusion_matrix_sum = np.zeros((len(unique_labels), len(unique_labels)))

    outer_accuracy = 0
    outer_conf_matrix_sum = 0
    for i in range(n_folds):
        test_dataset = folds[i]
        training_val_dataset = folds[:i] + folds[i + 1:]
        sum_accuracy = 0
        confusion_matrix_sum = np.zeros((len(unique_labels), len(unique_labels)))
        for j in range(len(training_val_dataset)):
            val_dataset = np.array(training_val_dataset[j])
            training_dataset = np.concatenate(training_val_dataset[:j] +
                                              training_val_dataset[j + 1:])
            tree, _ = decision_tree_learning(training_dataset, 0)
            while prune(val_dataset, tree):
                pass
            predictions, acc = evaluate(test_dataset, tree)
            sum_accuracy += acc
            conf_matrix = confusion_matrix(test_dataset[:, -1], predictions)
            confusion_matrix_sum += conf_matrix
        outer_accuracy += sum_accuracy / len(training_val_dataset)
        outer_conf_matrix_sum += confusion_matrix_sum / len(training_val_dataset)

    return outer_conf_matrix_sum / n_folds, outer_accuracy / n_folds


def prune(val_dataset: np.array, tree):
    def traverse(root, tree, val_dataset):
        """
        Prunes tree in place and returns a boolean indicating whether there are any nodes connected to 2 leaves left that will decrease validation error
        """
        if root is None or root['isLeaf']:
            return False

        left, right = None, None
        orig_acc = evaluate(val_dataset, tree)[1]
        if root['left']['isLeaf'] and root['right']['isLeaf']:
            left = root['left']
            right = root['right']
            root['isLeaf'] = True
            root['label'] = left['label']
            left_acc = evaluate(val_dataset, tree)[1]

            root['label'] = right['label']
            right_acc = evaluate(val_dataset, tree)[1]

            if (left_acc > right_acc and left_acc >= orig_acc):
                root['label'] = left['label']
                return True
            if (right_acc >= orig_acc):
                return True

            root['isLeaf'] = False
            return False

        first = traverse(root['left'], tree, val_dataset)
        second = traverse(root['right'], tree, val_dataset)

        return first or second

    return traverse(tree, tree, val_dataset)


def confusion_matrix(true_labels: np.array, predicted_labels: np.array):
    """
    Generate a confusion matrix based on predicted vs actual labels
    """
    assert len(true_labels) == len(predicted_labels)
    unique_labels = np.unique(np.concatenate((true_labels,
                                              predicted_labels)))

    confusion = np.zeros((len(unique_labels), len(unique_labels)))
    for true, predicted in zip(true_labels - 1, predicted_labels - 1):
        confusion[int(true), int(predicted)] += 1
    return confusion


def accuracy(true_labels, predicted_labels):
    """
    Calculate accuracy based on list of true and predicted labels
    """
    assert len(true_labels) == len(predicted_labels)

    correct_predictions = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            correct_predictions += 1

    return correct_predictions / len(true_labels)


def accuracy_from_confusion_matrix(confusion):
    """
    Calculate accuracy from confusion matrix
    """
    if np.sum(confusion) > 0:
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    else:
        return 0


def precision(confusion):
    """
    Calculate the precision matrix from the confusion matrix
    """
    precision_array = np.zeros((len(confusion),))

    for unique_label in range(confusion.shape[0]):
        if np.sum(confusion[:, unique_label]) > 0:
            precision_array[unique_label] = confusion[
                                                unique_label, unique_label] / np.sum(confusion[:,
                                                                                     unique_label])

    return precision_array


def recall(confusion):
    """
    Calculate the recall from the confusion matrix
    """
    recall_array = np.zeros((len(confusion),))

    for unique_label in range(confusion.shape[0]):
        if np.sum(confusion[unique_label, :]) > 0:
            recall_array[unique_label] = confusion[
                                             unique_label, unique_label] / np.sum(
                confusion[unique_label, :])

    return recall_array


def f1_score(precision_array, recall_array):
    """
    Calculate F1 score for each class given a precision and recall list 
    """
    assert len(precision_array) == len(recall_array)

    score_array = np.zeros((len(precision_array),))
    for label, (prec, rec) in enumerate(zip(precision_array, recall_array)):
        if prec + rec > 0:
            score_array[label] = 2 * prec * rec / (prec + rec)

    return score_array
