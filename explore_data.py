import pandas as pd

df = pd.read_csv('simulation_results/PhysionetMI/1.csv')
classifiers = df['Classifier'].unique()
print(f"Total classifiers: {len(classifiers)}")
for c in classifiers:
    if not c.startswith('subject'):
        print("BASELINE:", c)

print("\nAll subject_2 variations:")
for c in classifiers:
    if 'subject_2_' in c:
        print(c)
