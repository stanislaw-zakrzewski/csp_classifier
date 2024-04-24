import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

subjects = []
for subject_index in range(1, 10):
    subjects.append('s0{}.csv'.format(subject_index))
for subject_index in range(10, 53):
    subjects.append('s{}.csv'.format(subject_index))

methods = ['csp', 'parafac', 'combined']

accuracies_over_bands_data = None

try:
    accuracies_over_bands_data = pd.read_csv('accuracies_over_bands.csv')
except:
    print('File not found')

if accuracies_over_bands_data is None:
    data = {'method': [], 'subject': [], 'frequency': [], 'accuracy': []}
    for subject in subjects:
        best = {'method': '', 'accuracy': 0, 'frequency': 0}
        for method in methods:
            subject_data = pd.read_csv('ecai/data/{}/{}'.format(method, subject))
            averaged = subject_data.filter(items=['frequency', 'accuracy']).groupby('frequency',
                                                                                    as_index=False).mean()
            highest_accuracy = averaged.loc[averaged['accuracy'].idxmax()]
            if highest_accuracy['accuracy'] > best['accuracy']:
                best = {'method': method, 'accuracy': highest_accuracy['accuracy'],
                        'frequency': highest_accuracy['frequency']}
        if best['accuracy'] >= 0.6:
            data['method'].append(best['method'])
            data['subject'].append(subject)
            data['frequency'].append(best['frequency'])
            data['accuracy'].append(best['accuracy'])

    accuracies_over_bands_data = pd.DataFrame(data)
    accuracies_over_bands_data.to_csv('accuracies_over_bands.csv')

kmeans = KMeans(n_clusters=3).fit(
    accuracies_over_bands_data.drop(['Unnamed: 0', 'method', 'subject', 'accuracy'], axis=1))
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
unique, counts = np.unique(np.array(labels), return_counts=True)

a = np.array(accuracies_over_bands_data['subject'].tolist())
np.random.seed(23)
set_4 = np.random.choice(a, 20, replace=False)
set_5 = np.random.choice(set_4, 10, replace=False)
set_4 = list(set(set_4) - set(set_5))
set_4.sort()
set_5.sort()
print(set_4)
print(set_5)

groups = {}
for index, unique_entry in enumerate(unique):
    groups[unique_entry] = {'values': [], 'centroid': centroids[index][0]}

for index, row in accuracies_over_bands_data.iterrows():
    groups[labels[index]]['values'].append(row['subject'])

# print(groups)



# plt.scatter(accuracies_over_bands_data['x'], accuracies_over_bands_data['y'], c=kmeans.labels_.astype(float), s=50,
#             alpha=0.5)
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
# plt.show()

#
# scaler = StandardScaler()
# scaler.fit(accuracies_over_bands_data.drop(['method', 'subject', 'configuration', 'frequency'], axis=1))
# scaled_features = scaler.transform(accuracies_over_bands_data.drop(['method', 'subject', 'configuration', 'frequency'], axis=1))
