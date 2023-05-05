from matplotlib.figure import Figure
from matplotlib.pyplot import savefig
import seaborn as sns
from analyze_data import analyze_edf
from classifiers.parafac import process as parafacProcess
from classifiers.combined import process as combinedProcess
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from config_old import experiment_frequency_range, channels2

subjects = []
for i in range(9):
    subjects.append('s0{}'.format(i + 1))
for i in range(43):
    subjects.append('s{}'.format(i + 10))
# subjects = ['s14']
# experiment_frequency_range = (3, 4)

# print('CSP')
# for subject_index, subject in enumerate(subjects):
#     print(subject)
#     options = {
#         'configurations': [
#             {'channels': channels2, 'band_width': 1, 'step': 1},
#             {'channels': channels2, 'band_width': 3, 'step': 2},
#             {'channels': channels2, 'band_width': 6, 'step': 4}
#         ],
#         'memory': {'name': None, 'value': None},
#         'experiment_frequency_range': (4, 16)
#     }
#     filename = 'preprocessed_subjects/{}.edf'.format(subject)
#
#     accuracy_data = analyze_edf(filename, verbose='ERROR', options=options)
#     accuracy_data.to_csv('ecai/data/csp/{}.csv'.format(subject), index=False)
#
#     a = pd.read_csv('ecai/data/csp/{}.csv'.format(subject))
#     figure = Figure(figsize=(25, 10))
#     ax = figure.subplots()
#     sns.lineplot(data=a, x="frequency", y="accuracy", hue="configuration", style='configuration', errorbar=None, ax=ax,
#                  markers=True)
#
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
#     ax.grid()
#     figure.savefig('ecai/figures/csp/{}.png'.format(subject))

print('PARAFAC')
for subject_index, subject in enumerate(subjects):
    print(subject)
    options = {
        'configurations': [
            {'channels': channels2, 'band_width': 1, 'step': 1},
            {'channels': channels2, 'band_width': 3, 'step': 2},
            {'channels': channels2, 'band_width': 6, 'step': 4}
        ],
        'memory': {'name': None, 'value': None},
        'experiment_frequency_range': (4, 16)
    }
    filename = 'preprocessed_subjects/{}.edf'.format(subject)

    accuracy_data = analyze_edf(filename, classifier=parafacProcess, verbose='ERROR', options=options)
    accuracy_data.to_csv('ecai/data/parafac/{}.csv'.format(subject), index=False)

    a = pd.read_csv('ecai/data/parafac/{}.csv'.format(subject))
    figure = Figure(figsize=(25, 10))
    ax = figure.subplots()
    sns.lineplot(data=a, x="frequency", y="accuracy", hue="configuration", style='configuration', errorbar=None, ax=ax,
                 markers=True)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
    ax.grid()
    figure.savefig('ecai/figures/parafac/{}.png'.format(subject))

print('COMBINED')
for subject_index, subject in enumerate(subjects):
    print(subject)
    options = {
        'configurations': [
            {'channels': channels2, 'band_width': 1, 'step': 1},
            {'channels': channels2, 'band_width': 3, 'step': 2},
            {'channels': channels2, 'band_width': 6, 'step': 4}
        ],
        'memory': {'name': None, 'value': None},
        'experiment_frequency_range': (4, 16)
    }
    filename = 'preprocessed_subjects/{}.edf'.format(subject)

    accuracy_data = analyze_edf(filename, classifier=combinedProcess, verbose='ERROR', options=options)
    accuracy_data.to_csv('ecai/data/combined/{}.csv'.format(subject), index=False)

    a = pd.read_csv('ecai/data/combined/{}.csv'.format(subject))
    figure = Figure(figsize=(25, 10))
    ax = figure.subplots()
    sns.lineplot(data=a, x="frequency", y="accuracy", hue="configuration", style='configuration', errorbar=None, ax=ax,
                 markers=True)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
    ax.grid()
    figure.savefig('ecai/figures/combined/{}.png'.format(subject))

