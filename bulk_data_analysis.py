import pandas as pd
import numpy as np

dataframe = pd.read_csv('bulk_edf_data.csv')
dataframe.drop(dataframe[dataframe['date'].str.contains('2023-02-24T18-08-13')==True].index, inplace=True)

res = dataframe.groupby(['participant', 'task', 'date', 'classifier',
                         'frequency', 'configuration', 'frequency_start', 'frequency_end'], as_index=False).mean()

# Top 10% accuracy results for all of the participant/task/classifier combinations
res2 = res.groupby(['participant', 'task', 'classifier'], as_index=False)['accuracy'].apply(lambda grp: grp.nlargest(20).mean()).round(3)
res2.to_csv('res2.csv')

# CSP and PARAFAC accuracy difference
parafac = []
csp = []
for index, row in res2.iterrows():
    if row['classifier'] == 'csp':
        csp.append(row)
    else:
        parafac.append(row)

classifier_diff_data = {'participant': [], 'task': [], 'diff': []}
for csp_result in csp:
    parafac_result = []
    for p in parafac:
        if p['participant'] == csp_result['participant'] and p['task'] == csp_result['task']:
            parafac_result = p
            break

    classifier_diff_data['participant'].append(csp_result['participant'])
    classifier_diff_data['task'].append(csp_result['task'])
    classifier_diff_data['diff'].append(csp_result['accuracy'] - parafac_result['accuracy'])
res3 = pd.DataFrame(data=classifier_diff_data).round(3)
res3.to_csv('res3.csv')
# CSP and PARAFAC accuracy averaged over participants
res4 = res2.groupby(['task', 'classifier'], as_index=False)['accuracy'].mean().round(3)
res4.to_csv('res4.csv')
# Compare imagery to real combined
data_imagery_to_real = {'participant': [], 'type': [], 'accuracy': []}
res5 = res2.to_numpy()
temp_structure = {}
for res5row in res5:
    participant = res5row[0]
    if participant not in temp_structure:
        temp_structure[participant] = {}
    type = 'observation'
    if res5row[1][0:4] == 'imag':
        type = 'imagery'
    if res5row[1][0:4] == 'real':
        type = 'real'
    if type not in temp_structure[participant]:
        temp_structure[participant][type] = []
    temp_structure[participant][type].append(res5row[3])

for i in temp_structure:
    for j in temp_structure[i]:
        data_imagery_to_real['participant'].append(i)
        data_imagery_to_real['type'].append(j)
        data_imagery_to_real['accuracy'].append(np.average(temp_structure[i][j]))
res5 = pd.DataFrame(data=data_imagery_to_real).round(3)
res5.to_csv('res5.csv')
# for index, row in res2.iterrows():
#     accuracies =
# print(res)
