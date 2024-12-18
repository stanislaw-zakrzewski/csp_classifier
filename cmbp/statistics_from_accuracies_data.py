import pandas as pd
import seaborn as sns
from tkinter import filedialog as fd
import matplotlib.pyplot as plt

df = pd.read_csv('preprocessed_subjects_accuracies.csv')
df = pd.read_csv(fd.askopenfilename(filetypes=[("CSV data files", "*.csv")]))
pd.options.display.float_format = "{:,.2f}".format


threshold = .703 #dataset 1
# threshold = .65 # dataset 3_v2
# threshold = .8 # dataset tmp_3

max = df.groupby(['subject'], as_index=False)['accuracy'].max()
max = max[max['accuracy'] >= threshold]
max = max['subject'].values

min = df.groupby(['subject'], as_index=False)['accuracy'].max()
min = min[min['accuracy'] < threshold]
min = min['subject'].values

df_max = df[df['subject'].isin(max)]
df_min = df[df['subject'].isin(min)]

print('ALL')
print(f"All count: {len(df['subject'].unique())}")
print(df.groupby(['method'], as_index=False)['accuracy'].mean())
print(df.groupby(['method'], as_index=False)['percentage_of_data_used'].mean())
print('MIN')
print(f"Min count: {len(df_min['subject'].unique())}")
print(df_min.groupby(['method'], as_index=False)['accuracy'].mean())
print(df_min.groupby(['method'], as_index=False)['percentage_of_data_used'].mean())
print('MAX')
print(f"Max count: {len(df_max['subject'].unique())}")
print(df_max.groupby(['method'], as_index=False)['accuracy'].mean())
print(df_max.groupby(['method'], as_index=False)['percentage_of_data_used'].mean())

sns.set_theme(rc={'figure.figsize': (25, 10)})
ax = sns.barplot(x='subject', y='accuracy', hue='method', data=df_max)
for i in ax.containers:
    ax.bar_label(i, )
plt.show()