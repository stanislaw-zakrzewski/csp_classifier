import pandas as pd
import seaborn as sns
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
import numpy as np
import scipy



df = pd.read_csv('preprocessed_subjects_accuracies_old.csv')
df = pd.read_csv(fd.askopenfilename(filetypes=[("CSV data files", "*.csv")]))
df = df[df.method != "EEGNet"]
df = df[df.method != "EEGNet significant"]
pd.options.display.float_format = "{:,.2f}".format


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


# threshold = 0.755 # eegnet

# threshold = .703 #dataset 1 3 second
# threshold = .6685 #dataset 1 2 second
# threshold = .6 #dataset 1 1 second
# threshold = .695 #dataset 1 1 second offset
# threshold = .75 #dataset 1 128Hz
# threshold = .75 #dataset 1 256Hz
threshold = .65 # dataset 3_v2
# threshold = .8 # dataset tmp_3

max = df.groupby(['subject'], as_index=False)['accuracy'].max()
max = max[max['accuracy'] >= threshold]
max = max['subject'].values
# max = ['s03', 's04', 's14', 's15', 's23', 's35', 's37', 's41', 's43', 's48']
df_max = df[df['subject'].isin(max)]
print('MAX confidence intervals:')
for method in df_max['method'].unique():
    values = df_max.loc[df_max['method'] == method]['accuracy'].values
    print('\n',len(values), "{0:0.2f}".format(mean_confidence_interval(values)))
    print(method, "{0:0.2f}".format(np.mean(values)), ["{0:0.2f}".format(i) for i in sns.utils.ci(sns.algorithms.bootstrap(values))])

min = df.groupby(['subject'], as_index=False)['accuracy'].max()
min = min[min['accuracy'] < threshold]
min = min['subject'].values
# min = ['s01', 's02', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12', 's13', 's16',
#  's17', 's18', 's19', 's20', 's21', 's22', 's24', 's25', 's26', 's27', 's28', 's30',
#  's31', 's32', 's33', 's36', 's38', 's39', 's40', 's42', 's44', 's45', 's46', 's47',
#  's49', 's50', 's51', 's52']
df_min = df[df['subject'].isin(min)]
print('\nMIN confidence intervals:')
for method in df_min['method'].unique():
    values = df_min.loc[df_min['method'] == method]['accuracy'].values
    print('\n',len(values), "{0:0.2f}".format(mean_confidence_interval(values)))
    print(method, "{0:0.2f}".format(np.mean(values)), ["{0:0.2f}".format(i) for i in sns.utils.ci(sns.algorithms.bootstrap(values))])

print('\nALL confidence intervals:')
for method in df['method'].unique():
    values = df.loc[df['method'] == method]['accuracy'].values
    print('\n',len(values), "{0:0.2f}".format(mean_confidence_interval(values)))
    print(method, "{0:0.2f}".format(np.mean(values)), ["{0:0.2f}".format(i) for i in sns.utils.ci(sns.algorithms.bootstrap(values))])

ALPHA_NORMALITY = .05
methods = df['method'].unique()
print(methods)

def find_statistically_significant_differences(dataset, dataset_name):
    for i in range(len(methods)-1):
        for j in range(1,len(methods) - i):
            method_a = methods[i]
            method_b = methods[i+j]
            values_a = dataset.loc[dataset['method'] == method_a]['accuracy'].values
            values_b = dataset.loc[dataset['method'] == method_b]['accuracy'].values
            try:
                normality1 = scipy.stats.normaltest(values_a)[1]
                normality2 = scipy.stats.normaltest(values_b)[1]
                if normality1 < ALPHA_NORMALITY and normality2 < ALPHA_NORMALITY:
                    pvalue_less = scipy.stats.ttest_ind(values_a, values_b,
                                                  alternative='less').pvalue
                    pvalue_greater = scipy.stats.ttest_ind(values_a, values_b,
                                                     alternative='greater').pvalue
                else:
                    pvalue_less = scipy.stats.ranksums(values_a, values_b, alternative='less')[1]
                    pvalue_greater = \
                        scipy.stats.ranksums(values_a, values_b, alternative='greater')[1]
            except:
                pvalue_less = scipy.stats.ranksums(values_a, values_b, alternative='less')[1]
                pvalue_greater = \
                    scipy.stats.ranksums(values_a, values_b, alternative='greater')[1]
            if pvalue_less <= ALPHA_NORMALITY:
                print(f"For {dataset_name} {method_a} has less accuracy ({np.mean(values_a)}) than {method_b} ({np.mean(values_b)}) (pvalue = {pvalue_less})")
            if pvalue_greater <= ALPHA_NORMALITY:
                print(f"For {dataset_name} {method_a} has higher accuracy ({np.mean(values_a)}) than {method_b} ({np.mean(values_b)}) (pvalue = {pvalue_greater})")

find_statistically_significant_differences(df, 'ALL')
find_statistically_significant_differences(df_min, 'MIN')
find_statistically_significant_differences(df_max, 'MAX')
#
# print('ALL')
# print(f"All count: {len(df['subject'].unique())}")
# print(df.groupby(['method'], as_index=False)['accuracy'].mean())
# # print(df.groupby(['method'], as_index=False)['percentage_of_data_used_a'].mean())
# # print(df.groupby(['method'], as_index=False)['percentage_of_data_used_b'].mean())
# print(df.groupby(['method'], as_index=False)['time'].mean())
# print('MIN')
# print(f"Min count: {len(df_min['subject'].unique())}")
# print(df_min.groupby(['method'], as_index=False)['accuracy'].mean())
# # print(df_min.groupby(['method'], as_index=False)['percentage_of_data_used_a'].mean())
# # print(df_min.groupby(['method'], as_index=False)['percentage_of_data_used_b'].mean())
# print(df_min.groupby(['method'], as_index=False)['time'].mean())
# print('MAX')
# print(f"Max count: {len(df_max['subject'].unique())}")
# print(df_max.groupby(['method'], as_index=False)['accuracy'].mean())
# # print(df_max.groupby(['method'], as_index=False)['percentage_of_data_used_a'].mean())
# # print(df_max.groupby(['method'], as_index=False)['percentage_of_data_used_b'].mean())
# print(df_max.groupby(['method'], as_index=False)['time'].mean())
#
# sns.set_theme(rc={'figure.figsize': (25, 10)})
# ax = sns.barplot(x='subject', y='accuracy', hue='method', data=df_max, capsize=.2, errorbar='ci')
# for i in ax.containers:
#     ax.bar_label(i, )
# plt.show()
#
# ax = sns.barplot(x='method', y='accuracy', data=df_max, capsize=.2, errorbar='ci')
#
# plt.show()


