import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

items = np.load('significant_optimizer.npy', allow_pickle=True).item()
params = np.array(items['x_iters'])
values = items['func_vals']
order = values.argsort()
ordered_values = values[order]
ordered_params = params[order]
print((ordered_values-1) * -1)
print(ordered_params)
