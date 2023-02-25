import os
import pandas as pd
from progress.bar import Bar

from analyze_data import analyze_edf
from classifiers.flat import process as csp_classifier
from classifiers.parafac import process as parafac_classifier
from logger import ProgressBar
from visualization.accuracy_over_bands import save_visualized_accuracy_over_bands

print('BULK ANALYZE DATA STARTED')

directories = ['data_b', 'data_s']
classifiers = {'csp': csp_classifier, 'parafac': parafac_classifier}
directory_paths = {}
file_count = 0

for directory in directories:
    directory_paths[directory] = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            if filepath[-4:] == '.edf':
                directory_paths[directory].append(filepath)
                file_count += 1

data = []
progress_bar = ProgressBar('Processing files', max=file_count)

for directory_path in directory_paths:
    for filepath in directory_paths[directory_path]:
        for classifier in classifiers:

            edf_data = analyze_edf(filepath, classifier=classifiers[classifier], verbose='ERROR')
            columns = edf_data.columns
            date = filepath[7:26]
            task = filepath[27:-4]
            edf_data['participant'] = directory_path
            edf_data['task'] = task
            edf_data['date'] = date
            edf_data['classifier'] = classifier
            edf_data = edf_data[[*['participant', 'task', 'date', 'classifier'], *columns]]
            save_visualized_accuracy_over_bands(edf_data, '{}//graphs//{}_{}_{}.png'.format(directory_path, date, task,
                                                                                            classifier))
            if len(data) != 0:
                data = pd.concat([data, edf_data], ignore_index=True)
            else:
                data = edf_data
        progress_bar.next()

data.to_csv('bulk_edf_data.csv')
