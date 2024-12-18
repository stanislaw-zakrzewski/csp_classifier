from tkinter import filedialog as fd
from pyedflib import highlevel
import os

folder_path = fd.askdirectory()
annotation_counts = []
annotation_minimums = {}
annotation_maxes = {}
for file in os.listdir(folder_path):
    signals, signal_headers, header = highlevel.read_edf("{}/{}".format(folder_path, file))
    annotation_counts.append({})
    for annotation in header['annotations']:
        if annotation[2] in annotation_counts[-1]:
            annotation_counts[-1][annotation[2]] = annotation_counts[-1][annotation[2]] + 1
        else:
            annotation_counts[-1][annotation[2]] = 1
    for key in annotation_counts[-1]:
        if key in annotation_minimums:
            if annotation_minimums[key] > annotation_counts[-1][key]:
                annotation_minimums[key] = annotation_counts[-1][key]
        else:
            annotation_minimums[key] = annotation_counts[-1][key]
        if key in annotation_maxes:
            if annotation_maxes[key] < annotation_counts[-1][key]:
                annotation_maxes[key] = annotation_counts[-1][key]
        else:
            annotation_maxes[key] = annotation_counts[-1][key]

print(annotation_minimums)
print(annotation_maxes)
