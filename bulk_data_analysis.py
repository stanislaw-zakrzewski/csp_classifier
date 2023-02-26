import pandas as pd

dataframe = pd.read_csv('bulk_edf_data.csv')
# print(dataframe.count())
res = dataframe.groupby(['participant', 'task', 'date', 'classifier',
                         'frequency', 'configuration', 'frequency_start', 'frequency_end'], as_index=False).mean()

res2 = res.groupby(['participant', 'task', 'classifier'], as_index=False)['accuracy'].mean()

print(res)
