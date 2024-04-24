import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns



data = [
    {"test_subject_count": 1, "subset": "#1", "data": pd.read_csv('subset_accuracy_results_1_1_all.csv', index_col=0)},
    {"test_subject_count": 2, "subset": "#1", "data": pd.read_csv('subset_accuracy_results_1_2_all.csv', index_col=0)},
    {"test_subject_count": 1, "subset": "#2", "data": pd.read_csv('subset_accuracy_results_2_1_all.csv', index_col=0)},
    {"test_subject_count": 2, "subset": "#2", "data": pd.read_csv('subset_accuracy_results_2_2_all.csv', index_col=0)},
    {"test_subject_count": 1, "subset": "#4", "data": pd.read_csv('subset_accuracy_results_4_1_all.csv', index_col=0)},
    {"test_subject_count": 2, "subset": "#4", "data": pd.read_csv('subset_accuracy_results_4_2_all.csv', index_col=0)},
    {"test_subject_count": 1, "subset": "#5", "data": pd.read_csv('subset_accuracy_results_5_1_all.csv', index_col=0)},
    {"test_subject_count": 2, "subset": "#5", "data": pd.read_csv('subset_accuracy_results_5_2_all.csv', index_col=0)},
]

chart_data = {'Test subject count': [], 'Subset': [], 'Accuracy': []}
for data_entry in data:
    for accuracy in data_entry['data']['accuracy']:
        chart_data['Test subject count'].append(data_entry["test_subject_count"])
        chart_data['Subset'].append(data_entry['subset'])
        chart_data['Accuracy'].append(accuracy)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
g = sns.catplot(data=chart_data, x='Test subject count', y='Accuracy', hue='Subset', kind='bar', height=3, aspect=9/4)
g.set(ylim=(.4, .9))
plt.yticks([.4,.5,.6,.7,.8,.9])
plt.show()
