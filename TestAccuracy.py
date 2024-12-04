from SVM import run_SVM, run_SVM_Persuasion
from DecisionTree import run_DTC, run_DTC_Persuasion
from LogisticRegression import run_LR, run_LR_Persuasion
from CNN import run_CNN, run_CNN_Persuasion

gc_train_path = "gun_train_data.csv"
gc_test_path = "gun_test_data.csv"

ab_train_path = "abortion_train_data.csv"
ab_test_path = "abortion_test_data.csv"

gc_svm_accuracy, gc_svm_fp, gc_svm_fn = run_SVM(gc_train_path, gc_test_path)
gc_dtc_accuracy, gc_dtc_fp, gc_dtc_fn = run_DTC(gc_train_path, gc_test_path)
gc_cnn_accuracy, gc_cnn_fp, gc_cnn_fn = run_CNN(gc_train_path, gc_test_path)
gc_lr_accuracy, gc_lr_fp, gc_lr_fn = run_LR(gc_train_path, gc_test_path)

ab_svm_accuracy, ab_svm_fp, ab_svm_fn = run_SVM(ab_train_path, ab_test_path)
ab_dtc_accuracy, ab_dtc_fp, ab_dtc_fn = run_DTC(ab_train_path, ab_test_path)
ab_cnn_accuracy, ab_cnn_fp, ab_cnn_fn = run_CNN(ab_train_path, ab_test_path)
ab_lr_accuracy, ab_lr_fp, ab_lr_fn = run_LR(ab_train_path, ab_test_path)

gc_svm_persuasion_accuracy, gc_svm_persuasion_fp, gc_svm_persuasion_fn = run_SVM_Persuasion(gc_train_path, gc_test_path)
gc_dtc_persuasion_accuracy, gc_dtc_persuasion_fp, gc_dtc_persuasion_fn = run_DTC_Persuasion(gc_train_path, gc_test_path)
gc_cnn_persuasion_accuracy, gc_cnn_persuasion_fp, gc_cnn_persuasion_fn = run_CNN_Persuasion(gc_train_path, gc_test_path)
gc_lr_persuasion_accuracy, gc_lr_persuasion_fp, gc_lr_persuasion_fn = run_LR_Persuasion(gc_train_path, gc_test_path)

ab_svm_persuasion_accuracy, ab_svm_persuasion_fp, ab_svm_persuasion_fn = run_SVM_Persuasion(ab_train_path, ab_test_path)
ab_dtc_persuasion_accuracy, ab_dtc_persuasion_fp, ab_dtc_persuasion_fn = run_DTC_Persuasion(ab_train_path, ab_test_path)
ab_cnn_persuasion_accuracy, ab_cnn_persuasion_fp, ab_cnn_persuasion_fn = run_CNN_Persuasion(ab_train_path, ab_test_path)
ab_lr_persuasion_accuracy, ab_lr_persuasion_fp, ab_lr_persuasion_fn = run_LR_Persuasion(ab_train_path, ab_test_path)

print("Gun Control SVM Accuracy: ", gc_svm_accuracy)
print("Gun Control SVM False Positives: ", gc_svm_fp)
print("Gun Control SVM False Negatives: ", gc_svm_fn)

print("Gun Control DTC Accuracy: ", gc_dtc_accuracy)
print("Gun Control DTC False Positives: ", gc_dtc_fp)
print("Gun Control DTC False Negatives: ", gc_dtc_fn)

print("Gun Control CNN Accuracy: ", gc_cnn_accuracy)
print("Gun Control CNN False Positives: ", gc_cnn_fp)
print("Gun Control CNN False Negatives: ", gc_cnn_fn)

print("Gun Control LR Accuracy: ", gc_lr_accuracy)
print("Gun Control LR False Positives: ", gc_lr_fp)
print("Gun Control LR False Negatives: ", gc_lr_fn)


print("Abortion SVM Accuracy: ", ab_svm_accuracy)
print("Abortion SVM False Positives: ", ab_svm_fp)
print("Abortion SVM False Negatives: ", ab_svm_fn)

print("Abortion DTC Accuracy: ", ab_dtc_accuracy)
print("Abortion DTC False Positives: ", ab_dtc_fp)
print("Abortion DTC False Negatives: ", ab_dtc_fn)

print("Abortion CNN Accuracy: ", ab_cnn_accuracy)
print("Abortion CNN False Positives: ", ab_cnn_fp)
print("Abortion CNN False Negatives: ", ab_cnn_fn)

print("Abortion LR Accuracy: ", ab_lr_accuracy)
print("Abortion LR False Positives: ", ab_lr_fp)
print("Abortion LR False Negatives: ", ab_lr_fn)

svm_total_accuracy = (gc_svm_accuracy + ab_svm_accuracy) / 2
dtc_total_accuracy = (gc_dtc_accuracy + ab_dtc_accuracy) / 2
cnn_total_accuracy = (gc_cnn_accuracy + ab_cnn_accuracy) / 2
lr_total_accuracy = (gc_lr_accuracy + ab_lr_accuracy) / 2

print("Total SVM Accuracy: ", svm_total_accuracy)
print("Total DTC Accuracy: ", dtc_total_accuracy)
print("Total CNN Accuracy: ", cnn_total_accuracy)
print("Total LR Accuracy: ", lr_total_accuracy)


print("Gun Control SVM Persuasion Accuracy: ", gc_svm_persuasion_accuracy)
print("Gun Control SVM Persuasion False Positives: ", gc_svm_persuasion_fp)
print("Gun Control SVM Persuasion False Negatives: ", gc_svm_persuasion_fn)

print("Gun Control DTC Persuasion Accuracy: ", gc_dtc_persuasion_accuracy)
print("Gun Control DTC Persuasion False Positives: ", gc_dtc_persuasion_fp)
print("Gun Control DTC Persuasion False Negatives: ", gc_dtc_persuasion_fn)

print("Gun Control CNN Persuasion Accuracy: ", gc_cnn_persuasion_accuracy)
print("Gun Control CNN Persuasion False Positives: ", gc_cnn_persuasion_fp)
print("Gun Control CNN Persuasion False Negatives: ", gc_cnn_persuasion_fn)

print("Gun Control LR Persuasion Accuracy: ", gc_lr_persuasion_accuracy)
print("Gun Control LR Persuasion False Positives: ", gc_lr_persuasion_fp)
print("Gun Control LR Persuasion False Negatives: ", gc_lr_persuasion_fn)


print("Abortion SVM Persuasion Accuracy: ", ab_svm_persuasion_accuracy)
print("Abortion SVM Persuasion False Positives: ", ab_svm_persuasion_fp)
print("Abortion SVM Persuasion False Negatives: ", ab_svm_persuasion_fn)

print("Abortion DTC Persuasion Accuracy: ", ab_dtc_persuasion_accuracy)
print("Abortion DTC Persuasion False Positives: ", ab_dtc_persuasion_fp)    
print("Abortion DTC Persuasion False Negatives: ", ab_dtc_persuasion_fn)

print("Abortion CNN Persuasion Accuracy: ", ab_cnn_persuasion_accuracy)
print("Abortion CNN Persuasion False Positives: ", ab_cnn_persuasion_fp)
print("Abortion CNN Persuasion False Negatives: ", ab_cnn_persuasion_fn)

print("Abortion LR Persuasion Accuracy: ", ab_lr_persuasion_accuracy)
print("Abortion LR Persuasion False Positives: ", ab_lr_persuasion_fp)
print("Abortion LR Persuasion False Negatives: ", ab_lr_persuasion_fn)


svm_persuasion_total_accuracy = (gc_svm_persuasion_accuracy + ab_svm_persuasion_accuracy) / 2
dtc_persuasion_total_accuracy = (gc_dtc_persuasion_accuracy + ab_dtc_persuasion_accuracy) / 2
cnn_persuasion_total_accuracy = (gc_cnn_persuasion_accuracy + ab_cnn_persuasion_accuracy) / 2
lr_persuasion_total_accuracy = (gc_lr_persuasion_accuracy + ab_lr_persuasion_accuracy) / 2

print("Total SVM Persuasion Accuracy: ", svm_persuasion_total_accuracy)
print("Total DTC Persuasion Accuracy: ", dtc_persuasion_total_accuracy)
print("Total CNN Persuasion Accuracy: ", cnn_persuasion_total_accuracy)