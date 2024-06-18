import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy import stats

accuracies_comparison = []
classifiers = ["RandomForestClassifier", "LogisticRegression", "KNeighborsClassifier",
               "LinearDiscriminantAnalysis", "GaussianNB", "SVC", "DecisionTreeClassifier",
               "MLPClassifier", "ExtraTreesClassifier", "AdaBoostClassifier",
               "GradientBoostingClassifier", "XGBClassifier"]
short_name = ["RF", "LR", "KNN", "LDA", "GNB", "SVM", "DT", "MLP", "ET", "AB", "GB", "XGB"]

# SINGLE CLASSIFIERS
accuracy_values = []
mean_accuracy_stdev = []
mean_accuracy = []
stdev = []
mean_specificity = []
mean_sensitivity = []

df_single_classifiers = pd.read_csv(f"performance_metrics/single_classifiers.csv")
for classifier in classifiers:
    df_byclassifier = df_single_classifiers[df_single_classifiers["classifier"] == classifier]
    accuracy_values.append((df_byclassifier["accuracy"].values).tolist())
    mean_accuracy.append(f"{round(np.mean(df_byclassifier['accuracy']) * 100, 2)}")
    stdev.append(f"±{round(np.std(df_byclassifier['accuracy']) * 100, 2)}")
    mean_accuracy_stdev.append(f"{round(np.mean(df_byclassifier['accuracy']) * 100, 2)} (±{round(np.std(df_byclassifier['accuracy']) * 100, 2)})")
    mean_specificity.append(f"{round(np.mean(df_byclassifier['specificity']) * 100, 2)}")
    mean_sensitivity.append(f"{round(np.mean(df_byclassifier['sensitivity']) * 100, 2)}")

accuracies_comparison.append(mean_accuracy)

plt.figure(figsize=(10, 6))
sns.set_palette(["#6bc4ff"])
sns.boxplot(x='classifier', y='accuracy', data=df_single_classifiers)
plt.title('Accuracies of single-classifier models')
plt.xticks(range(len(short_name)), short_name)
plt.savefig("analysis/boxplot_single_classifiers.png")
# plt.show()

df_single_classifiers = pd.DataFrame([mean_accuracy_stdev, mean_specificity, mean_sensitivity])
df_single_classifiers.columns = short_name
df_single_classifiers.index = ["Accuracy", "Specificity", "Sensitivity"]

print("SINGLE CLASSIFIERS (TABLE 6 Original)")
print(df_single_classifiers)
print(df_single_classifiers.to_latex())

friedman = stats.friedmanchisquare(*accuracy_values)
print(friedman)  # esto es para todos
if friedman.pvalue < 0.05:
    accuracy_values = np.array(accuracy_values)
    nemenyi = sp.posthoc_nemenyi_friedman(accuracy_values.T)
    nemenyi.columns = short_name
    nemenyi.index = short_name
    nemenyi = nemenyi.round(decimals=4)
    print(nemenyi)
    nemenyi.to_csv('analysis/nemenyi_single.csv')
    print(nemenyi.to_latex())

print(f"-----------------------------------------------------------------------\n")


# ALL_MODELS BY TASK
df_metrics_bytask = pd.read_csv("performance_metrics/allmodels_bytask_metrics.csv")
df_accuracies_bytask = pd.DataFrame(columns=['Task #'] + short_name)
for task in range(1, 20):
    mean_accuracy_bytask = []
    for classifier in classifiers:
        df = df_metrics_bytask[(df_metrics_bytask["classifier"] == classifier) & (df_metrics_bytask["task"] == task)]
        mean_accuracy_bytask.append(f"{round(np.mean(df['accuracy']) * 100, 2)} (±{round(np.std(df['accuracy']) * 100, 2)})")
    df_accuracies_bytask.loc[len(df_accuracies_bytask)] = [task] + mean_accuracy_bytask

df_accuracies_bytask.set_index('Task #', inplace=True)

print("BYTASK CLASSIFIERS ACCURACY (TABLE 7 Original)")
print(df_accuracies_bytask)
print(df_accuracies_bytask.to_latex())


print("Mean by task:")
accuracies_bytask = []
for task in range(1, 20):
    df = df_metrics_bytask[df_metrics_bytask["task"] == task]
    accuracies_bytask.append(df['accuracy'].to_list())

print(accuracies_bytask)

friedman = stats.friedmanchisquare(*accuracies_bytask)
print(friedman)  # esto es para todos
if friedman.pvalue < 0.05:
    accuracies_bytask = np.array(accuracies_bytask)
    nemenyi = sp.posthoc_nemenyi_friedman(accuracies_bytask.T)
    nemenyi.columns = range(1, 20)
    nemenyi.index = range(1, 20)
    nemenyi = nemenyi.round(decimals=4)
    print(nemenyi)
    print(nemenyi.to_latex())

plt.figure(figsize=(10, 6))
sns.set_palette(["#ffcb59"])
sns.boxplot(x='task', y='accuracy', data=df_metrics_bytask)
plt.title('Accuracies of task-specific models by tasks')
plt.savefig("analysis/boxplot_task_specific_bytasks.png")
# plt.show()


print("Mean by classifier:")
accuracies_byclassifier = []
for classifier in classifiers:
    df = df_metrics_bytask[df_metrics_bytask["classifier"] == classifier]
    accuracies_byclassifier.append(df['accuracy'].to_list())

print(accuracies_byclassifier)

friedman = stats.friedmanchisquare(*accuracies_byclassifier)
print(friedman)
if friedman.pvalue < 0.05:
    accuracies_byclassifier = np.array(accuracies_byclassifier)
    nemenyi = sp.posthoc_nemenyi_friedman(accuracies_byclassifier.T)
    nemenyi.columns = short_name
    nemenyi.index = short_name
    nemenyi = nemenyi.round(decimals=4)
    print(nemenyi)
    print(nemenyi.to_latex())

plt.figure(figsize=(10, 6))
sns.set_palette(["#ffcb59"])
sns.boxplot(x='classifier', y='accuracy', data=df_metrics_bytask)
plt.title('Accuracies of task-specific models by classifier')
plt.xticks(range(len(short_name)), short_name)
plt.savefig("analysis/boxplot_task_specific_byclassifier.png")
# plt.show()

list_accuracies_bytask = []
for task in range(1, 20):
    df = df_metrics_bytask[(df_metrics_bytask["task"] == task)]
    accuracy_values = (df["accuracy"].values).tolist()
    list_accuracies_bytask.append(accuracy_values)
friedman = stats.friedmanchisquare(*list_accuracies_bytask)
print(friedman)

friedman_pvalues = []
for task in range(1, 20):
    list_accuracies_bytask = []
    for classifier in classifiers:
        df = df_metrics_bytask[(df_metrics_bytask["classifier"] == classifier) & (df_metrics_bytask["task"] == task)]
        accuracy_values = (df["accuracy"].values).tolist()
        list_accuracies_bytask.append(accuracy_values)
    friedman = stats.friedmanchisquare(*list_accuracies_bytask)
    if friedman[1] > 0.05:
        print(f"Task {task} Classifiers: {round(friedman[1], 4)} - No Significative")
    else:
        print(f"Task {task} Classifiers: {round(friedman[1], 4)} - Significative")

print(f"-----------------------------------------------------------------------\n")


# BYTASK CLASSIFIERS Sp & Se
short_classifiers = ["RF_Sp", "RF_Se", "LR_Sp", "LR_Se", "KNN_Sp", "KNN_Se", "LDA_Sp", "LDA_Se",
                     "GNB_Sp", "GNB_Se", "SVM_Sp", "SVM_Se", "DT_Sp", "DT_Se", "MLP_Sp", "MLP_Se",
                     "ET_Sp", "ET_Se", "AB_Sp", "AB_Se", "GB_Sp", "GB_Se", "XGB_Sp", "XGB_Se"]

df_metrics_bytask = pd.read_csv("performance_metrics/allmodels_bytask_metrics.csv")
df_spec_sens_bytask = pd.DataFrame(columns=['Task #'] + short_classifiers)
for task in range(1, 20):
    mean_specificity = []
    mean_sensitivity = []
    for classifier in classifiers:
        df = df_metrics_bytask[(df_metrics_bytask["classifier"] == classifier) & (df_metrics_bytask["task"] == task)]
        mean_specificity.append(f"{round(np.mean(df['specificity']) * 100, 2)}")
        mean_sensitivity.append(f"{round(np.mean(df['sensitivity']) * 100, 2)}")
    df_spec_sens_bytask.loc[len(df_spec_sens_bytask)] = [task] + [val for pair in zip(mean_specificity, mean_sensitivity) for val in pair]

df_spec_sens_bytask.set_index('Task #', inplace=True)

print("BYTASK CLASSIFIERS Sp & Se (TABLE 8 Original)")
print(df_spec_sens_bytask)
print(df_spec_sens_bytask.to_latex())

print(f"-----------------------------------------------------------------------\n")
print("SINGLE & MIXED CLASSIFIER TASK ESPECIFIC MODELS (TABLE 9 Original)")

# Read CSVs
df_single_classifiers_task_specific = pd.read_csv(f"performance_metrics/single_classifier_task_specific.csv")
df_mixed_classifiers_task_specific = pd.read_csv(f"performance_metrics/mixed_classifier_task_specific.csv")

# Process dataframes
accuracy_values = []
mean_accuracy_stdev = []
mean_accuracy = []
stdev = []
mean_specificity = []
mean_sensitivity = []

# Process individual classifiers
for classifier in classifiers:
    df_byclassifier = df_single_classifiers_task_specific[df_single_classifiers_task_specific["classifier"] == classifier]
    classifier_accuracy_values = (df_byclassifier["accuracy"].values).tolist()
    accuracy_values.append(classifier_accuracy_values)
    mean_accuracy.append(f"{round(np.mean(df_byclassifier['accuracy']) * 100, 2)}")
    stdev.append(f"±{round(np.std(df_byclassifier['accuracy']) * 100, 2)}")
    mean_accuracy_stdev.append(f"{round(np.mean(df_byclassifier['accuracy']) * 100, 2)} (±{round(np.std(df_byclassifier['accuracy']) * 100, 2)})")
    mean_specificity.append(f"{round(np.mean(df_byclassifier['specificity']) * 100, 2)}")
    mean_sensitivity.append(f"{round(np.mean(df_byclassifier['sensitivity']) * 100, 2)}")

# Process mixed classifiers
df_mixed_classifiers_task_specific['classifier'] = 'MIXED'
classifier_accuracy_values = (df_mixed_classifiers_task_specific["accuracy"].values).tolist()
accuracy_values.append(classifier_accuracy_values)
mean_accuracy.append(f"{round(np.mean(df_mixed_classifiers_task_specific['accuracy']) * 100, 2)}")
stdev.append(f"±{round(np.std(df_mixed_classifiers_task_specific['accuracy']) * 100, 2)}")
mean_accuracy_stdev.append(f"{round(np.mean(df_mixed_classifiers_task_specific['accuracy']) * 100, 2)} (±{round(np.std(df_mixed_classifiers_task_specific['accuracy']) * 100, 2)})")
mean_specificity.append(f"{round(np.mean(df_mixed_classifiers_task_specific['specificity']) * 100, 2)}")
mean_sensitivity.append(f"{round(np.mean(df_mixed_classifiers_task_specific['sensitivity']) * 100, 2)}")

# Combine dataframes
df_combined = pd.concat([df_single_classifiers_task_specific, df_mixed_classifiers_task_specific], ignore_index=True)

accuracies_comparison.append(mean_accuracy)

# Create Boxplot
plt.figure(figsize=(10, 6))
sns.set_palette(["#f35a33"])
sns.boxplot(x='classifier', y='accuracy', data=df_combined)
plt.title('Accuracies combining task specific classifiers')
plt.xticks(range(len(short_name) + 1), (short_name + ["MIX"]))
plt.savefig("analysis/boxplot_combining_task_specific.png")
# plt.show()

# Create dataframe with aggregated metrics
df_single_classifiers = pd.DataFrame([mean_accuracy_stdev, mean_specificity, mean_sensitivity])
df_single_classifiers.columns = (short_name + ["MIX"])
df_single_classifiers.index = ["Accuracy", "Specificity", "Sensitivity"]

print(df_single_classifiers)
print(df_single_classifiers.to_latex())

friedman = stats.friedmanchisquare(*accuracy_values)
print(friedman)
if friedman.pvalue < 0.05:
    accuracy_values = np.array(accuracy_values)
    nemenyi = sp.posthoc_nemenyi_friedman(accuracy_values.T)
    nemenyi.columns = (short_name + ["MIX"])
    nemenyi.index = (short_name + ["MIX"])
    nemenyi = nemenyi.round(decimals=4)
    print(nemenyi)
    print(nemenyi.to_latex())


print(f"-----------------------------------------------------------------------\n")
print("Top 5 Single & Mixed Classifiers Models:")

df_top5_single_classifiers = pd.read_csv("performance_metrics/top5_single_classifiers_voting_models.csv")
df_top5_mixed_classifiers = pd.read_csv("performance_metrics/top5_mixed_classifiers_voting_models.csv")

accuracy_values = []
mean_accuracy_stdev = []
mean_accuracy = []
stdev = []
mean_specificity = []
mean_sensitivity = []

# Process individual classifiers
for classifier in classifiers:
    df_byclassifier = df_top5_single_classifiers[df_top5_single_classifiers["classifier"] == classifier]
    classifier_accuracy_values = df_byclassifier["accuracy"].values.tolist()
    accuracy_values.append(classifier_accuracy_values)
    mean_accuracy.append(round(np.mean(df_byclassifier["accuracy"]) * 100, 2))
    stdev.append(round(np.std(df_byclassifier["accuracy"]) * 100, 2))
    mean_accuracy_stdev.append(f"{mean_accuracy[-1]} (±{stdev[-1]})")
    mean_specificity.append(round(np.mean(df_byclassifier["specificity"]) * 100, 2))
    mean_sensitivity.append(round(np.mean(df_byclassifier["sensitivity"]) * 100, 2))


df_top5_mixed_classifiers['classifier'] = 'MIXED'
classifier_accuracy_values = (df_top5_mixed_classifiers["accuracy"].values).tolist()
accuracy_values.append(classifier_accuracy_values)
mean_accuracy.append(f"{round(np.mean(df_top5_mixed_classifiers['accuracy']) * 100, 2)}")
stdev.append(f"±{round(np.std(df_top5_mixed_classifiers['accuracy']) * 100, 2)}")
mean_accuracy_stdev.append(f"{round(np.mean(df_top5_mixed_classifiers['accuracy']) * 100, 2)} (±{round(np.std(df_mixed_classifiers_task_specific['accuracy']) * 100, 2)})")
mean_specificity.append(f"{round(np.mean(df_top5_mixed_classifiers['specificity']) * 100, 2)}")
mean_sensitivity.append(f"{round(np.mean(df_top5_mixed_classifiers['sensitivity']) * 100, 2)}")

accuracies_comparison.append(mean_accuracy)

# Combine dataframes
df_combined = pd.concat([df_top5_single_classifiers, df_top5_mixed_classifiers], ignore_index=True)

plt.figure(figsize=(10, 6))
sns.set_palette(["#6bc96c"])
sns.boxplot(x='classifier', y='accuracy', data=df_combined)
plt.title('Accuracies combining Top 5 Single Classifiers')
plt.xticks(range(len(short_name) + 1), (short_name + ["MIX"]))
plt.ylim(0.55, 1.0)
plt.savefig("analysis/boxplot_top5_single_classifiers.png")
# plt.show()

# Create dataframe with aggregated metrics
df_top5_mixed = pd.DataFrame([mean_accuracy_stdev, mean_specificity, mean_sensitivity])
df_top5_mixed.columns = (short_name + ["MIX"])
df_top5_mixed.index = ["Accuracy", "Specificity", "Sensitivity"]

print(df_top5_mixed)
print(df_top5_mixed.to_latex())

friedman = stats.friedmanchisquare(*accuracy_values)
print(friedman)
if friedman.pvalue < 0.05:
    accuracy_values = np.array(accuracy_values)
    nemenyi = sp.posthoc_nemenyi_friedman(accuracy_values.T)
    nemenyi.columns = (short_name + ["MIX"])
    nemenyi.index = (short_name + ["MIX"])
    nemenyi = nemenyi.round(decimals=4)
    print(nemenyi)
    nemenyi.to_csv('analysis/nemenyi_single.csv')
    print(nemenyi.to_latex())


print(f"-----------------------------------------------------------------------\n")
print("Top 3 Single & Mixed Classifiers Models:")

df_top3_single_classifiers = pd.read_csv("performance_metrics/top3_single_classifiers_voting_models.csv")
df_top3_mixed_classifiers = pd.read_csv("performance_metrics/top3_mixed_classifiers_voting_models.csv")

accuracy_values = []
mean_accuracy_stdev = []
mean_accuracy = []
stdev = []
mean_specificity = []
mean_sensitivity = []

# Process individual classifiers
for classifier in classifiers:
    df_byclassifier = df_top3_single_classifiers[df_top3_single_classifiers["classifier"] == classifier]
    classifier_accuracy_values = df_byclassifier["accuracy"].values.tolist()
    accuracy_values.append(classifier_accuracy_values)
    mean_accuracy.append(round(np.mean(df_byclassifier["accuracy"]) * 100, 2))
    stdev.append(round(np.std(df_byclassifier["accuracy"]) * 100, 2))
    mean_accuracy_stdev.append(f"{mean_accuracy[-1]} (±{stdev[-1]})")
    mean_specificity.append(round(np.mean(df_byclassifier["specificity"]) * 100, 2))
    mean_sensitivity.append(round(np.mean(df_byclassifier["sensitivity"]) * 100, 2))

# Add mixed classifier with appropriate name
df_top3_mixed_classifiers['classifier'] = 'MIX'
classifier_accuracy_values = df_top3_mixed_classifiers["accuracy"].values.tolist()
accuracy_values.append(classifier_accuracy_values)
mean_accuracy.append(round(np.mean(df_top3_mixed_classifiers['accuracy']) * 100, 2))
stdev.append(round(np.std(df_top3_mixed_classifiers['accuracy']) * 100, 2))
mean_accuracy_stdev.append(f"{mean_accuracy[-1]} (±{stdev[-1]})")
mean_specificity.append(round(np.mean(df_top3_mixed_classifiers['specificity']) * 100, 2))
mean_sensitivity.append(round(np.mean(df_top3_mixed_classifiers['sensitivity']) * 100, 2))

accuracies_comparison.append(mean_accuracy)

# Combine dataframes
df_combined = pd.concat([df_top3_single_classifiers, df_top3_mixed_classifiers], ignore_index=True)

plt.figure(figsize=(10, 6))
sns.set_palette(["#7be87d"])
sns.boxplot(x='classifier', y='accuracy', data=df_combined)
plt.title('Accuracies combining Top 3 Single Classifiers')
plt.xticks(range(len(short_name) + 1), short_name + ["MIX"])
plt.ylim(0.55, 1.0)
plt.savefig("analysis/boxplot_top3_single_classifiers.png")
# plt.show()

# Create dataframe with aggregated metrics
df_top3_mixed = pd.DataFrame({
    "Accuracy": mean_accuracy_stdev,
    "Specificity": mean_specificity,
    "Sensitivity": mean_sensitivity
}, index=short_name + ["MIX"])

df_top3_mixed = df_top3_mixed.T
print(df_top3_mixed)
print(df_top3_mixed.to_latex())

friedman = stats.friedmanchisquare(*accuracy_values)
print(friedman)  # esto es para todos
if friedman.pvalue < 0.05:
    accuracy_values = np.array(accuracy_values)
    nemenyi = sp.posthoc_nemenyi_friedman(accuracy_values.T)
    nemenyi.columns = short_name + ["MIX"]
    nemenyi.index = short_name + ["MIX"]
    nemenyi = nemenyi.round(decimals=4)
    print(nemenyi)
    print(nemenyi.to_latex())

print(f"-----------------------------------------------------------------------\n")
print("Comparison against the original")

# Originals
single_classifiers_orig = [88.29, 81.86, 71.43, 72.14, 85.00, 79.00, 78.57, 83.14]
combining_task_specific_orig = [88.57, 85.71, 85.71, 77.14, 82.86, 88.57, 94.29, 88.57, None, None, None, None, 91.43]

accuracies_comparison = [single_classifiers_orig] + [combining_task_specific_orig] + accuracies_comparison

df_comparison = pd.DataFrame(accuracies_comparison)
df_comparison.index = ["single_classifiers_orig", "combining_task_specific_orig",
                       "single_classifiers", "combining_task_specific",
                       "top5_single_classifiers", "top3_single_classifiers"]
df_comparison.columns = short_name + ["MIX"]
df_comparison.to_csv("analysis/df_comparison.csv")
print(df_comparison.to_latex())

# Table Data
data = {
    'Classifier': ['RF', 'LR', 'KNN', 'LDA', 'GNB', 'SVM', 'DT', 'MLP', 'ET', 'AB', 'GB', 'XGB', 'MIX'],
    'single_classifiers_orig': [88.29, 81.86, 71.43, 72.14, 85, 79, 78.57, 83.14, None, None, None, None, None],
    'combining_task_specific_orig': [88.57, 85.71, 85.71, 77.14, 82.86, 88.57, 94.29, 88.57, None, None, None, None, 91.43],
    'single_classifiers': [87.61, 83.75, 67.16, 83.07, 86.93, 83.98, 71.48, 83.41, 87.05, 84.09, 84.89, 83.98, None],
    'combining_task_specific': [84.2, 78.07, 78.98, 78.41, 75.23, 78.52, 81.36, 80.68, 83.3, 84.55, 83.86, 83.18, 85],
    'top5_single_classifiers': [87.39, 84.89, 67.73, 83.07, 86.93, 85.57, 80.11, 84.55, 88.64, 85.68, 86.36, 84.2, 87.27],
    'top3_single_classifiers': [87.27, 84.43, 67.73, 83.07, 86.93, 85.45, 72.95, 85.23, 88.75, 85.23, 85.91, 84.2, 88.52]
}

# Create a dataframe of the data
df = pd.DataFrame(data)

# Melt the dataframe to facilitate graphing with seaborn
df_melted = df.melt(id_vars='Classifier', var_name='Group', value_name='Accuracy')

# Personalized color palette
palette = {
    'single_classifiers_orig': '#4b4e6c',
    'combining_task_specific_orig': '#ed0f87',
    'single_classifiers': '#189bcc',
    'combining_task_specific': '#f35a33',
    'top5_single_classifiers': '#eed959',
    'top3_single_classifiers': '#3abc12'
}

# Create the bar chart in portrait orientation
plt.figure(figsize=(10, 14))
sns.barplot(data=df_melted, y='Classifier', x='Accuracy', hue='Group', palette=palette)
plt.xlim(60, 100)  # Establecer el límite del eje x de 60 a 100

# Add vertical lines
for x in range(65, 101, 5):
    plt.axvline(x=x, color='#dbdfdf', linestyle='--', linewidth=0.5)

plt.title('Comparison of Accuracies of All Model Types')
plt.xlabel('Accuracy (%)')
plt.ylabel('Classifiers')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("analysis/barplot_comparison_models.png")
plt.show()

# Significant differences between originals and ours
means_combiningtaskspecific_orig = np.array([88.57, 85.71, 85.71, 77.14, 82.86, 88.57, 94.29, 88.57, 91.43])
means_combiningtaskspecific = np.array([84.20, 78.07, 78.98, 78.41, 75.23, 78.52, 81.36, 80.68, 85.00])

wilcoxon_test = stats.wilcoxon(means_combiningtaskspecific_orig, means_combiningtaskspecific)

print(wilcoxon_test)

means_single_classifiers_orig = np.array([88.29, 81.86, 71.43, 72.14, 85.00, 79.00, 78.57, 83.14])
means_single_classifiers = np.array([87.61, 83.75, 67.16, 83.07, 86.93, 83.98, 71.48, 83.41])

wilcoxon_test = stats.wilcoxon(means_single_classifiers_orig, means_single_classifiers)

print(wilcoxon_test)


