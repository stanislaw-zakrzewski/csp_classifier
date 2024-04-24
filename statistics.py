import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data_1 = pd.read_csv('pp-rai-2024/subset_accuracy_results_1_1_all.csv')
data_2 = pd.read_csv('pp-rai-2024/subset_accuracy_results_2_1_all.csv')
data_3 = pd.read_csv('pp-rai-2024/subset_accuracy_results_4_1_all.csv')
# data_4 = pd.read_csv('subset_accuracy_results_5_2_all.csv')

normality_test_pvalue = {'dataset': [], 'test_subjects': [], 'pvalue': []}
stat_values = {}
all_accuracies = {}
for dataset_id in [1, 2, 4, 5]:
    stat_values[dataset_id] = {}
    all_accuracies[dataset_id] = {}
    for test_subject_count in [1, 2]:
        normality_test_pvalue['dataset'].append(dataset_id)
        normality_test_pvalue['test_subjects'].append(test_subject_count)
        accuracies = list(
            pd.read_csv('subset_accuracy_results_{}_{}_all.csv'.format(dataset_id, test_subject_count))['accuracy'])
        all_accuracies[dataset_id][test_subject_count] = accuracies
        pvalue = stats.normaltest(accuracies)
        normality_test_pvalue['pvalue'].append(pvalue[1])
        stat_values[dataset_id][test_subject_count] = pvalue[1]

normality_test_pvalue = pd.DataFrame(normality_test_pvalue)
normality_test_pvalue_matrix = normality_test_pvalue.pivot_table(index='dataset', columns='test_subjects',
                                                                 values='pvalue')
ax = sns.heatmap(normality_test_pvalue_matrix, cmap="YlGnBu", annot=True, fmt="0.5f")
ax.set(ylabel='Dataset', xlabel='Subjects in test set')
plt.show()

tests_to_run = [
    {'subjects': [1, 4], 'test_subjects': 1},
    {'subjects': [1, 4], 'test_subjects': 2},
    {'subjects': [1, 5], 'test_subjects': 1},
    {'subjects': [1, 5], 'test_subjects': 2},
    {'subjects': [2, 4], 'test_subjects': 1},
    {'subjects': [2, 4], 'test_subjects': 2},
    {'subjects': [2, 5], 'test_subjects': 1},
    {'subjects': [2, 5], 'test_subjects': 2},
]

for test_to_run in tests_to_run:
    subject1, subject2 = test_to_run['subjects']
    test_subject_count = test_to_run['test_subjects']
    accuracies1 = all_accuracies[subject1][test_subject_count]
    accuracies2 = all_accuracies[subject2][test_subject_count]
    max_pvalue = max(stat_values[subject1][test_subject_count], stat_values[subject2][test_subject_count])
    # Non normal distribution
    if max_pvalue > .05:
        pvalue_less = stats.ranksums(accuracies1, accuracies2, alternative='less')
        pvalue_greater = stats.ranksums(accuracies1, accuracies2, alternative='greater')
        test_to_run['test'] = 'ranksums'
    # Normal distribution
    else:
        pvalue_less = stats.ttest_ind(accuracies1, accuracies2, alternative='less')
        pvalue_greater = stats.ttest_ind(accuracies1, accuracies2, alternative='greater')
        test_to_run['test'] = 'ttest'
    test_to_run['average_accuracies'] = [np.average(accuracies1), np.average(accuracies2)]
    test_to_run['less'] = pvalue_less[1]
    test_to_run['greater'] = pvalue_greater[1]

tests_with_statistical_significance = []
for test_to_run in tests_to_run:
    for alternative in ['less', 'greater']:
        if test_to_run[alternative] <= .05:
            tests_with_statistical_significance.append({
                'test': test_to_run['test'],
                'alternative': alternative,
                'pvalue': test_to_run[alternative],
                'subjects': test_to_run['subjects'],
                'test_subjects': test_to_run['test_subjects'],
                'average_accuracies': test_to_run['average_accuracies'],
            })

for test_with_statistical_significance in tests_with_statistical_significance:
    print(test_with_statistical_significance)



# acc_1 = list(data_1['accuracy'])
# acc_2 = list(data_2['accuracy'])
# acc_3 = list(data_3['accuracy'])
#
# print(stats.ranksums(acc_1, acc_3, alternative='greater'))

# print(stats.normaltest(acc_1))
# print(stats.normaltest(acc_2))
# print(stats.normaltest(acc_3))
#
# print(stats.ttest_ind(acc_1, acc_2, equal_var=True, axis=0))
