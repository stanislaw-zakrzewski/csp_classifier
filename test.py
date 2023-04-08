# Matplotlib
# https://www.geeksforgeeks.org/python-basic-gantt-chart-using-matplotlib/
# https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.broken_barh.html

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

source = pd.DataFrame([
    {"drama": "Pride and Prejudice", "start": '1795-01-01', "end": '1810-01-01'},
    {"drama": "Sense and Sensibility", "start": '1792-01-01', "end": '1797-01-01'},
    {"drama": "Jane Eyre", "start": '1799-01-01', "end": '1819-01-01'},
    {"drama": "Bridgerton", "start": '1813-01-01', "end": '1827-01-01'},
    {"drama": "Middlemarch", "start": '1829-01-01', "end": '1832-01-01'},
    {"drama": "Cranford", "start": '1842-01-01', "end": '1843-01-01'},
    {"drama": "David Copperfield", "start": '1840-01-01', "end": '1860-01-01'},
    {"drama": "Poldark", "start": '1781-01-01', "end": '1801-01-01'},
    {"drama": "North and South", "start": '1850-01-01', "end": '1860-01-01'},
    {"drama": "Barchester Chronicles", "start": '1855-01-01', "end": '1867-02-01'},
    {"drama": "The Way We Live Now", "start": '1870-01-01', "end": '1880-02-01'},
    {"drama": "Tess of the D’Urbervilles", "start": '1880-01-01', "end": '1890-02-01'},
    {"drama": "Upstairs, Downstairs", "start": '1903-01-01', "end": '1930-02-01'},
    {"drama": "Downton Abbey", "start": '1912-01-01', "end": '1939-02-01'},
    {"drama": "Jewel in the Crown", "start": '1942-01-01', "end": '1947-02-01'},
    {"drama": "Poldark", "start": '1957-01-01', "end": '1967-02-01'},

])

source['start'] = pd.to_datetime(source['start'])
source['end'] = pd.to_datetime(source['end'])
source['diff'] = source['end'] - source['start']

# Declaring a figure "gnt"
fig, gnt = plt.subplots(figsize=(8, 6))

# Need to fix hidden tick labels
# https://stackoverflow.com/questions/43673659/matplotlib-not-showing-first-label-on-x-axis-for-the-bar-plot

y_tick_labels = source.drama.values
y_pos = np.arange(len(y_tick_labels))

gnt.set_yticks(y_pos)
gnt.set_yticklabels(y_tick_labels)

# https://sparkbyexamples.com/python/iterate-over-rows-in-pandas-dataframe/
# https://www.tutorialspoint.com/plotting-dates-on-the-x-axis-with-python-s-matplotlib
# https://matplotlib.org/stable/gallery/color/named_colors.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
# https://www.geeksforgeeks.org/how-to-annotate-matplotlib-scatter-plots
for index, row in source.sort_values(by='start').reset_index().iterrows():
    start_year = int(row.start.strftime("%Y"))
    duration = row['diff'].days / 365
    gnt.broken_barh([(start_year, duration)],
                    (index - 0.5, 0.8),
                    facecolors=('tan'),
                    label=row.drama)
    gnt.text(start_year + 0.5, index - 0.2, row.drama)
plt.show()