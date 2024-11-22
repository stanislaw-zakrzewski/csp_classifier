import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
import pandas as pd

df_path = fd.askopenfilename()
df = pd.read_csv(df_path)

sns.set_theme(rc={'figure.figsize': (25, 10)})
ax = sns.barplot(x='subject', y='accuracy', hue='method', data=df)
for i in ax.containers:
    ax.bar_label(i, )
plt.show()