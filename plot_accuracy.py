import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracies(csv_file, window_size=1):
    # Load the data
    df = pd.read_csv(csv_file)
    
    if window_size > 1:
        # Ensure data is sorted by Classifier and Trial for rolling calculation
        df = df.sort_values(by=['Classifier', 'Trial'])
        df['Cumulative_Accuracy'] = df.groupby('Classifier')['Cumulative_Accuracy'].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
    
    # Extract Classifier Type
    def get_classifier_type(name):
        if "CSP_LDA" in name:
            return "CSP + LDA"
        elif "Cov_Tangent" in name:
            return "Cov + Tangent Space + LR"
        elif "CSP_SVM" in name:
            return "CSP + SVM"
        return "Other"
        
    df['Classifier_Type'] = df['Classifier'].apply(get_classifier_type)
    
    # Shorten names for labels (e.g. subject_1_... -> S1)
    def get_short_name(name):
        is_static = name.endswith("_static")
        clean_name = name[:-7] if is_static else name
        
        if clean_name.startswith("subject_"):
            short = f"S{clean_name.split('_')[1]}"
        elif clean_name.startswith("baseline_"):
            short = "Baseline"
        else:
            short = clean_name
            
        return f"{short} (Static)" if is_static else short
        
    df['Short_Name'] = df['Classifier'].apply(get_short_name)
    
    # Create the faceted plot
    g = sns.relplot(
        data=df, 
        x='Trial', 
        y='Cumulative_Accuracy', 
        hue='Short_Name', 
        col='Classifier_Type',
        kind='line',
        alpha=0.7, 
        linewidth=1.5,
        legend=False,
        errorbar=None,
        height=6,
        aspect=1.0,
        facet_kws={'sharey': True}
    )
    
    # Formatting the plot
    g.fig.suptitle('Cumulative Accuracy Over Trials by Classifier Type', fontsize=16, y=1.05)
    g.set_axis_labels('Trial', 'Cumulative Accuracy')
    
    # Format each subplot
    for ax in g.axes.flat:
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Clean up subplot title
        title = ax.get_title().replace('Classifier_Type = ', '')
        ax.set_title(title, fontsize=14)
        
        # Add text to the lines
        df_type = df[df['Classifier_Type'] == title]
        for short_name in df_type['Short_Name'].unique():
            clf_data = df_type[df_type['Short_Name'] == short_name]
            if not clf_data.empty:
                last_point = clf_data.iloc[-1]
                
                # Highlight Baseline
                fw = 'bold' if short_name == 'Baseline' else 'normal'
                color = '#ff0000' if short_name == 'Baseline' else '#333333'
                
                ax.text(last_point['Trial'] + 0.5, 
                         last_point['Cumulative_Accuracy'], 
                         short_name, 
                         fontsize=9,
                         color=color,
                         fontweight=fw,
                         va='center')

        # Expand x-axis to make room for labels (since labels are short now, 15% is enough)
        max_trial = df_type['Trial'].max()
        if pd.notna(max_trial):
            ax.set_xlim(0, max_trial + (max_trial * 0.15))
            
    plt.tight_layout()
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    # csv_filename = "adaptive_simulation_Cho2017_sub14.csv"
    csv_filename = "simulation_results/PhysionetMI/7.csv"
    window_size = 5  # Moving average window size (set to 1 to show original chart)
    print(f"Loading data from {csv_filename} and generating plot (moving average window = {window_size})...")
    plot_accuracies(csv_filename, window_size=window_size)
