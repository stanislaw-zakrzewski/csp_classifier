import pandas as pd

df = pd.read_csv('preprocessed_subjects_csp_accuracies.csv')
print(df.groupby(['method'])['accuracy'].mean())