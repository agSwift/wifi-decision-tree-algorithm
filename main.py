from utils.dataImport import load_noisy_dataset, load_clean_dataset
from utils.evaluation import perform_cross_validation, perform_nested_cross_validation, precision, recall, f1_score
from utils.visualization import show_entire_clean_tree

if __name__ == '__main__':
    # Clean dataset results with no pruning
    print("Clean dataset results with no pruning: ")
    clean_dataset = load_clean_dataset()
    avg_no_pruning_confusion_matrix, no_pruning_accuracy = perform_cross_validation(clean_dataset, n_folds=10)
    recall_no_pruning_array = recall(avg_no_pruning_confusion_matrix)
    precision_no_pruning_array = precision(avg_no_pruning_confusion_matrix)
    score_array = f1_score(precision_no_pruning_array, recall_no_pruning_array)
    print("Confusion Matrix: \n", avg_no_pruning_confusion_matrix, )
    print("Accuracy: ", no_pruning_accuracy)
    print("Recall: ", recall_no_pruning_array)
    print("Precision: ", precision_no_pruning_array)
    print("F1-Score: ", score_array)
    print()
    print()

    # Noisy dataset results with no pruning
    print("Noisy dataset results with no pruning: ")
    noisy_dataset = load_noisy_dataset()
    avg_no_pruning_confusion_matrix, no_pruning_accuracy = perform_cross_validation(noisy_dataset, n_folds=10)
    recall_no_pruning_array = recall(avg_no_pruning_confusion_matrix)
    precision_no_pruning_array = precision(avg_no_pruning_confusion_matrix)
    score_array = f1_score(precision_no_pruning_array, recall_no_pruning_array)
    print("Confusion Matrix: \n", avg_no_pruning_confusion_matrix, )
    print("Accuracy: ", no_pruning_accuracy)
    print("Recall: ", recall_no_pruning_array)
    print("Precision: ", precision_no_pruning_array)
    print("F1-Score: ", score_array)
    print()
    print()

    # Clean dataset results with pruning
    print("Clean dataset results with pruning: ")
    clean_dataset = load_clean_dataset()
    avg_pruning_confusion_matrix, pruning_accuracy = perform_nested_cross_validation(clean_dataset, n_folds=10)
    recall_pruning_array = recall(avg_pruning_confusion_matrix)
    precision_pruning_array = precision(avg_pruning_confusion_matrix)
    score_array = f1_score(precision_pruning_array, recall_pruning_array)
    print("Confusion Matrix: \n", avg_pruning_confusion_matrix, )
    print("Accuracy: ", pruning_accuracy)
    print("Recall: ", recall_pruning_array)
    print("Precision: ", precision_pruning_array)
    print("F1-Score: ", score_array)
    print()
    print()

    # Noisy dataset results with pruning
    print("Noisy dataset results with pruning: ")
    noisy_dataset = load_noisy_dataset()
    avg_pruning_confusion_matrix, pruning_accuracy = perform_nested_cross_validation(noisy_dataset, n_folds=10)
    recall_pruning_array = recall(avg_pruning_confusion_matrix)
    precision_pruning_array = precision(avg_pruning_confusion_matrix)
    score_array = f1_score(precision_pruning_array, recall_pruning_array)
    print("Confusion Matrix: \n", avg_pruning_confusion_matrix, )
    print("Accuracy: ", pruning_accuracy)
    print("Recall: ", recall_pruning_array)
    print("Precision: ", precision_pruning_array)
    print("F1-Score: ", score_array)
    print()
    print()

    # Visualisation of decision tree for entire clean dataset
    show_entire_clean_tree()
