from SVM import run_SVM
from DecisionTree import run_DTC
from LogisticRegression import run_LR
# from CNN import run_CNN

gc_train_path = "gun_train_data.csv"
gc_test_path = "gun_test_data.csv"

ab_train_path = "abortion_train_data.csv"
ab_test_path = "abortion_test_data.csv"

gc_svm_accuracy, gc_svm_fp, gc_svm_fn = run_SVM(gc_train_path, gc_test_path)
gc_dtc_accuracy, gc_dtc_fp, gc_dtc_fn = run_DTC(gc_train_path, gc_test_path)
# gc_cnn_accuracy, gc_cnn_fp, gc_cnn_fn = run_CNN(gc_train_path, gc_test_path)
gc_lr_accuracy, gc_lr_fp, gc_lr_fn = run_LR(gc_train_path, gc_test_path)

# ab_svm_accuracy, ab_svm_fp, ab_svm_fn = run_SVM(ab_train_path, ab_test_path)
# ab_dtc_accuracy, ab_dtc_fp, ab_dtc_fn = run_DTC(ab_train_path, ab_test_path)
# ab_cnn_accuracy, ab_cnn_fp, ab_cnn_fn = run_CNN(ab_train_path, ab_test_path)
# ab_lr_accuracy, ab_lr_fp, ab_lr_fn = run_LR(ab_train_path, ab_test_path)

print("Gun Control SVM Accuracy: ", gc_svm_accuracy)
print("Gun Control SVM False Positives: ", gc_svm_fp)
print("Gun Control SVM False Negatives: ", gc_svm_fn)

print("Gun Control DTC Accuracy: ", gc_dtc_accuracy)
print("Gun Control DTC False Positives: ", gc_dtc_fp)
print("Gun Control DTC False Negatives: ", gc_dtc_fn)

# print("Gun Control CNN Accuracy: ", gc_cnn_accuracy)
# print("Gun Control CNN False Positives: ", gc_cnn_fp)
# print("Gun Control CNN False Negatives: ", gc_cnn_fn)

print("Gun Control LR Accuracy: ", gc_lr_accuracy)
print("Gun Control LR False Positives: ", gc_lr_fp)
print("Gun Control LR False Negatives: ", gc_lr_fn)


# print("Abortion SVM Accuracy: ", ab_svm_accuracy)
# print("Abortion SVM False Positives: ", ab_svm_fp)
# print("Abortion SVM False Negatives: ", ab_svm_fn)

# print("Abortion DTC Accuracy: ", ab_dtc_accuracy)
# print("Abortion DTC False Positives: ", ab_dtc_fp)
# print("Abortion DTC False Negatives: ", ab_dtc_fn)

# print("Abortion CNN Accuracy: ", ab_cnn_accuracy)
# print("Abortion CNN False Positives: ", ab_cnn_fp)
# print("Abortion CNN False Negatives: ", ab_cnn_fn)

# print("Abortion LR Accuracy: ", ab_lr_accuracy)
# print("Abortion LR False Positives: ", ab_lr_fp)
# print("Abortion LR False Negatives: ", ab_lr_fn)