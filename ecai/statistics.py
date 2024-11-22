from os import walk
import os
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np

paths = [
    {'model': 'csp', 'path': 'ecai\\data\\csp'},
    {'model': 'parafac', 'path': 'ecai\\data\\parafac'},
    {'model': 'combined', 'path': 'ecai\\data\\combined'}
]

generate_max_for_all = False
top_10_percent_for_subjects = False
best_frequency = False
best_frequency2 = False
heatmap_only_1hz = False
heatmap_different_frequencies = False
histograms_frequencies=False
bandwidth_heatmap = False
best_overall = True

all_data = None
try:
    all_data = pd.read_csv('all_data.csv')
except:
    pass

if all_data is None:
    for path in paths:
        absolute_path = os.path.join(os.getcwd(), path['path'])
        csv_files = os.listdir(absolute_path)

        for csv_file in csv_files:
            data = pd.read_csv(os.path.join(absolute_path, csv_file), index_col=False)
            data['subject'] = csv_file[1:3]
            data['classifier'] = path['model']
            if all_data is not None:
                all_data = pd.concat((all_data, data))
            else:
                all_data = data
    all_data.reset_index()
    all_data.to_csv('all_data.csv')

if generate_max_for_all:
    res = all_data.groupby(['subject', 'classifier', 'configuration'], as_index=False)['accuracy'].max()

    sns.lineplot(data=res, x="subject", y="accuracy", hue="classifier", style='classifier', errorbar=None,
                 markers=True)

    plt.show()

if top_10_percent_for_subjects:
    def categorize(x):
        np_x = np.array(x)
        a = np_x[np_x.argsort()[-int(len(x) / 10):][::-1]]
        return a.mean()


    res = all_data.groupby(['subject', 'classifier'], as_index=False)['accuracy'].agg([categorize])
    print(res)

    sns.lineplot(data=res, x="subject", y="categorize", hue="classifier", style='classifier', errorbar=None,
                 markers=True)

    plt.show()

if best_frequency:
    def categorize(x, y):
        np_x = np.array(x)
        a = np_x[np_x.argsort()[-int(len(x) / 10):][::-1]]
        return a.mean()


    res = all_data.groupby(['subject', 'configuration'], as_index=False)


    def wavg(group):
        d = np.array(group['frequency'])
        w = np.array(group['accuracy'])
        # return d[np.argmax(w)]

        k = {}
        k['best_frequency'] = d[np.argmax(w)]
        k['accuracy_for_frequency'] = w[np.argmax(w)]
        # d['b_mean'] = x['b'].mean()
        # d['c_d_prodsum'] = (x['c'] * x['d']).sum()
        return pd.Series(k, index=['best_frequency', 'accuracy_for_frequency'])


    res = res.apply(wavg)
    # res = res.rename(columns={None: "best_frequency"})
    print(res)

    sns.scatterplot(data=res, x="subject", y="best_frequency", hue="accuracy_for_frequency", style='configuration',
                    markers=True)
    plt.grid()
    plt.show()

if best_frequency2:
    def categorize(x, y):
        np_x = np.array(x)
        a = np_x[np_x.argsort()[-int(len(x) / 10):][::-1]]
        return a.mean()


    res = all_data.groupby(['subject', 'frequency'], as_index=False)['accuracy'].max()
    print(res)
    sns.heatmap(all_data.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'))
    plt.show()


    def wavg(group):
        d = np.array(group['frequency'])
        w = np.array(group['accuracy'])
        # return d[np.argmax(w)]

        k = {}
        k['best_frequency'] = d[np.argmax(w)]
        k['accuracy_for_frequency'] = w[np.argmax(w)]
        # d['b_mean'] = x['b'].mean()
        # d['c_d_prodsum'] = (x['c'] * x['d']).sum()
        return pd.Series(k, index=['best_frequency', 'accuracy_for_frequency'])


    res = res.apply(wavg)
    # res = res.rename(columns={None: "best_frequency"})
    print(res)

    sns.scatterplot(data=res, x="subject", y="best_frequency", hue="accuracy_for_frequency",
                    markers=True)
    plt.grid()
    plt.show()

if heatmap_only_1hz:
    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    # res = res.drop(res[(res.configuration == '3Hz width 11 channels') | (res.configuration == '6Hz width 11 channels')].index)
    res = res.drop(res[(res.classifier == 'combined') | (res.classifier == 'parafac')].index)
    # res = res.drop(res[res.accuracy < .7].index)
    res2 = res.groupby(['subject'], as_index=False)['accuracy'].max()
    print('CSP:', round(res2['accuracy'].mean() * 100.0, 2))

    sns.heatmap(res.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'), cmap="crest",
                vmin=.5, vmax=.9)
    plt.title('Accuracy results using CSP based model.')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('csp_accuracies.csv.png', dpi=200, bbox_inches='tight')
    plt.clf()

    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    # res = res.drop(res[(res.configuration == '1Hz width 11 channels') | (res.configuration == '6Hz width 11 channels')].index)
    res = res.drop(res[(res.classifier == 'combined') | (res.classifier == 'csp')].index)
    # res = res.drop(res[res.accuracy < .7].index)
    res2 = res.groupby(['subject'], as_index=False)['accuracy'].max()
    print('PARAFAC:', round(res2['accuracy'].mean() * 100.0, 2))

    sns.heatmap(res.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'), cmap="crest",
                vmin=.5, vmax=.9)
    plt.title('Accuracy results using PARAFAC based model.')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('parafac_accuracies.png', dpi=200, bbox_inches='tight')
    plt.clf()

    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    # res = res.drop(res[(res.configuration == '1Hz width 11 channels') | (res.configuration == '3Hz width 11 channels')].index)
    res = res.drop(res[(res.classifier == 'parafac') | (res.classifier == 'csp')].index)
    # res = res.drop(res[res.accuracy < .7].index)
    res2 = res.groupby(['subject'], as_index=False)['accuracy'].max()
    print('Combined:', round(res2['accuracy'].mean() * 100.0, 2))

    sns.heatmap(res.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'), cmap="crest",
                vmin=.5, vmax=.9)
    plt.title('Accuracy results using combined model.')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('combined_accuracies.png', dpi=200, bbox_inches='tight')
    plt.clf()

if heatmap_different_frequencies:
    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    res = res.drop(
        res[(res.configuration == '3Hz width 11 channels') | (res.configuration == '6Hz width 11 channels')].index)
    res = res.drop(res[(res.classifier == 'combined') | (res.classifier == 'parafac')].index)
    res1 = res.groupby(['subject'], as_index=False)['accuracy'].max()
    print('CSP:', round(res1['accuracy'].mean() * 100.0, 2))

    # res = res.drop(res[res.accuracy < .7].index)

    sns.heatmap(res.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'), cmap="crest",
                vmin=.5, vmax=.9)
    plt.title('Accuracy results using CSP based model for 1Hz width frequency bands.')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('1hz_csp.png', dpi=200, bbox_inches='tight')
    plt.clf()

    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    res = res.drop(
        res[(res.configuration == '1Hz width 11 channels') | (res.configuration == '6Hz width 11 channels')].index)
    res = res.drop(res[(res.classifier == 'combined') | (res.classifier == 'parafac')].index)
    # res = res.drop(res[res.accuracy < .7].index)
    res2 = res.groupby(['subject'], as_index=False)['accuracy'].max()
    print('PARAFAC:', round(res2['accuracy'].mean() * 100.0, 2))

    sns.heatmap(res.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'), cmap="crest",
                vmin=.5, vmax=.9)
    plt.title('Accuracy results using CSP based model for 3Hz width frequency bands.')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('3hz_csp.png', dpi=200, bbox_inches='tight')
    plt.clf()

    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    res = res.drop(
        res[(res.configuration == '1Hz width 11 channels') | (res.configuration == '3Hz width 11 channels')].index)
    res = res.drop(res[(res.classifier == 'combined') | (res.classifier == 'parafac')].index)
    # res = res.drop(res[res.accuracy < .7].index)
    res3 = res.groupby(['subject'], as_index=False)['accuracy'].max()
    print('Combined:', round(res3['accuracy'].mean() * 100.0, 2))

    sns.heatmap(res.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'), cmap="crest",
                vmin=.5, vmax=.9)
    plt.title('Accuracy results using CSP based model for 6Hz width frequency bands.')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('6hz_csp.png', dpi=200, bbox_inches='tight')
    plt.clf()

if histograms_frequencies:
    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    res = res.drop(
        res[(res.configuration == '3Hz width 11 channels') | (res.configuration == '6Hz width 11 channels')].index)
    res1 = res.groupby(['subject', 'classifier'], as_index=False)['accuracy'].max()
    res1 = res1.groupby(['classifier'], as_index=False)['accuracy'].mean()
    res1['accuracy'] = res1['accuracy'].apply(lambda x: round(x * 100.0, 2))
    print('1Hz')
    print(res1)
    sns.histplot(res, x="accuracy", hue="classifier", element="poly", cumulative=True)
    plt.title('Accuracy results using CSP based model for 1Hz width frequency bands.')
    plt.gcf().set_size_inches(10, 5)
    plt.xlim([.35, .9])
    plt.savefig('1hz_csp.png', dpi=200, bbox_inches='tight')
    plt.clf()

    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    res = res.drop(
        res[(res.configuration == '1Hz width 11 channels') | (res.configuration == '6Hz width 11 channels')].index)
    res1 = res.groupby(['subject', 'classifier'], as_index=False)['accuracy'].max()
    res1 = res1.groupby(['classifier'], as_index=False)['accuracy'].mean()
    res1['accuracy'] = res1['accuracy'].apply(lambda x: round(x * 100.0, 2))
    print('3Hz')
    print(res1)
    sns.histplot(res, x="accuracy", hue="classifier", element="poly", cumulative=True)
    plt.title('Accuracy results using CSP based model for 3Hz width frequency bands.')
    plt.gcf().set_size_inches(10, 5)
    plt.xlim([.35, .9])
    plt.savefig('3hz_csp.png', dpi=200, bbox_inches='tight')
    plt.clf()

    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    res = res.drop(
        res[(res.configuration == '1Hz width 11 channels') | (res.configuration == '3Hz width 11 channels')].index)
    res1 = res.groupby(['subject', 'classifier'], as_index=False)['accuracy'].max()
    res1 = res1.groupby(['classifier'], as_index=False)['accuracy'].mean()
    res1['accuracy'] = res1['accuracy'].apply(lambda x: round(x * 100.0, 2))
    print('6Hz')
    print(res1)
    sns.histplot(res, x="accuracy", hue="classifier", element="poly", cumulative=True)
    plt.title('Accuracy results using CSP based model for 6Hz width frequency bands.')
    plt.gcf().set_size_inches(10, 5)
    plt.xlim([.35, .9])
    plt.savefig('6hz_csp.png', dpi=200, bbox_inches='tight')
    plt.clf()

if bandwidth_heatmap:
    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    res = res.drop(res[(res.configuration == '3Hz width 11 channels') | (res.configuration == '6Hz width 11 channels')].index)
    res = res.drop(res[(res.classifier == 'combined') | (res.classifier == 'parafac')].index)
    # res = res.drop(res[res.accuracy < .7].index)
    res2 = res.groupby(['subject'], as_index=False)['accuracy'].max()
    print('CSP:', round(res2['accuracy'].mean() * 100.0, 2))

    sns.heatmap(res.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'), cmap="crest",
                vmin=.5, vmax=.9)
    plt.title('Accuracy results using CSP based model.')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('1hz_heatmap.png', dpi=200, bbox_inches='tight')
    plt.clf()

    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    res = res.drop(res[(res.configuration == '1Hz width 11 channels') | (res.configuration == '6Hz width 11 channels')].index)
    res = res.drop(res[(res.classifier == 'combined') | (res.classifier == 'parafac')].index)
    # res = res.drop(res[res.accuracy < .7].index)
    res2 = res.groupby(['subject'], as_index=False)['accuracy'].max()
    print('PARAFAC:', round(res2['accuracy'].mean() * 100.0, 2))

    sns.heatmap(res.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'), cmap="crest",
                vmin=.5, vmax=.9)
    plt.title('Accuracy results using CSP based model.')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('3hz_heatmap.png', dpi=200, bbox_inches='tight')
    plt.clf()

    res = all_data.drop(all_data[(all_data.subject == 29) | (all_data.subject == 34)].index)
    res = res.drop(res[(res.configuration == '1Hz width 11 channels') | (res.configuration == '3Hz width 11 channels')].index)
    res = res.drop(res[(res.classifier == 'combined') | (res.classifier == 'parafac')].index)
    # res = res.drop(res[res.accuracy < .7].index)
    res2 = res.groupby(['subject'], as_index=False)['accuracy'].max()
    print('Combined:', round(res2['accuracy'].mean() * 100.0, 2))

    sns.heatmap(res.pivot_table(index='frequency', columns='subject', values='accuracy', aggfunc='max'), cmap="crest",
                vmin=.5, vmax=.9)
    plt.title('Accuracy results using CSP model.')
    plt.gcf().set_size_inches(10, 5)
    plt.savefig('6hz_heatmap.png', dpi=200, bbox_inches='tight')
    plt.clf()

if best_overall:
    res = all_data.groupby(['subject', 'classifier', 'configuration'], as_index=False)['accuracy'].max()
    res = res.loc[res.groupby('subject')['accuracy'].idxmax()]
    print(res['configuration'].value_counts())
    print(res.groupby(['configuration'], as_index=False)['accuracy'].mean())
    # print(res)

