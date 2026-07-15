import os
import glob
import pandas as pd

def categorize_classifier(name):
    # Determine base type
    if "CSP_LDA" in name:
        base_type = "CSP + LDA"
    elif "Cov_Tangent" in name:
        base_type = "Cov + Tangent Space + LR"
    elif "CSP_SVM" in name:
        base_type = "CSP + SVM"
    else:
        base_type = "Unknown"
        
    # Determine variant
    if name.startswith("baseline_"):
        variant = "baseline"
    elif name.endswith("_static"):
        variant = "static"
    else:
        variant = "adaptive"
        
    return f"{base_type} ({variant})"

# Adjustable variable to select only the top N percent of static and adaptive classifiers
# Set between 0.0 and 1.0 (e.g., 0.1 for top 10%). Set to 1.0 to include all.
TOP_PERCENT = 0.1

def main():
    directory = os.path.normpath("simulation_results/PhysionetMI")
    print(f"Reading CSV files from: {directory}")
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("averaged_accuracies") and not os.path.basename(f).startswith("classifier_ranking")]
    
    if not csv_files:
        print("No CSV files found.")
        return
        
    all_final_accuracies = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            idx = df.groupby('Classifier')['Trial'].idxmax()
            final_accs = df.loc[idx, ['Classifier', 'Cumulative_Accuracy']].copy()
            final_accs['Source_File'] = os.path.basename(file)
            all_final_accuracies.append(final_accs)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    if not all_final_accuracies:
        print("No data extracted.")
        return
        
    combined_df = pd.concat(all_final_accuracies, ignore_index=True)
    
    # Calculate top 10 mean accuracy per classifier
    classifier_stats = combined_df.groupby('Classifier').agg(
        Average_Accuracy=('Cumulative_Accuracy', lambda x: x.nlargest(10).mean()),
        Files_Count=('Source_File', 'count')
    ).reset_index()
    
    # Categorize
    classifier_stats['Type'] = classifier_stats['Classifier'].apply(categorize_classifier)
    
    # Filter static and adaptive categories to only keep the top N percent of classifiers
    if TOP_PERCENT < 1.0:
        filtered_stats_list = []
        for type_name, group in classifier_stats.groupby('Type'):
            if "baseline" in type_name.lower():
                # Keep all baseline classifiers
                filtered_stats_list.append(group)
            else:
                # Sort descending by Average_Accuracy and keep top TOP_PERCENT
                sorted_group = group.sort_values(by='Average_Accuracy', ascending=False)
                n_keep = max(1, int(round(len(sorted_group) * TOP_PERCENT)))
                filtered_stats_list.append(sorted_group.head(n_keep))
        filtered_stats = pd.concat(filtered_stats_list, ignore_index=True)
    else:
        filtered_stats = classifier_stats.copy()
        
    # Rank overall categories by average of their selected classifiers
    category_summary = filtered_stats.groupby('Type').agg(
        Category_Average_Accuracy=('Average_Accuracy', 'mean'),
        Classifiers_Count=('Classifier', 'count')
    ).reset_index().sort_values(by='Category_Average_Accuracy', ascending=False)
    
    print("\n--- Category Summary ---")
    if TOP_PERCENT < 1.0:
        print(f"(Averaged over top {TOP_PERCENT * 100:.0f}% of static and adaptive classifiers)")
    print(category_summary.to_string(index=False))
    
    # Create the report content
    report_lines = []
    report_lines.append("# Classifier Ranking Report")
    report_lines.append("")
    report_lines.append("This report ranks the classifiers evaluated on the PhysionetMI dataset by their type.")
    report_lines.append("For each classifier, the average accuracy is computed using its **top 10 best runs** across all simulation files.")
    if TOP_PERCENT < 1.0:
        report_lines.append(f"Static and adaptive categories are filtered to include only the **top {TOP_PERCENT * 100:.0f}%** of classifiers in that category.")
    report_lines.append("")
    report_lines.append("## Category Ranking")
    report_lines.append("")
    report_lines.append("| Rank | Classifier Type | Overall Average Accuracy | Classifiers in Category |")
    report_lines.append("|---|---|---|---|")
    for i, row in enumerate(category_summary.itertuples(), 1):
        report_lines.append(f"| {i} | **{row.Type}** | {row.Category_Average_Accuracy:.6f} | {row.Classifiers_Count} |")
    report_lines.append("")
    
    # For each category, show details and ranking of the top classifiers
    report_lines.append("## Detailed Classifier Rankings by Type")
    report_lines.append("")
    
    for row in category_summary.itertuples():
        report_lines.append(f"### {row.Type}")
        report_lines.append("")
        if "baseline" not in row.Type.lower() and TOP_PERCENT < 1.0:
            report_lines.append(f"*Showing the top {TOP_PERCENT * 100:.0f}% of classifiers in this category.*")
            report_lines.append("")
        report_lines.append("| Rank | Classifier Name | Average Accuracy (Top 10) | Total Runs |")
        report_lines.append("|---|---|---|---|")
        
        cat_df = filtered_stats[filtered_stats['Type'] == row.Type].sort_values(by='Average_Accuracy', ascending=False)
        # Limit to top 15 or so for readability if it's large, but baseline has only 1, so show all if small
        limit = 15
        for rank, c_row in enumerate(cat_df.itertuples(), 1):
            if rank > limit:
                report_lines.append(f"| ... | *and {len(cat_df) - limit} more classifiers* | | |")
                break
            report_lines.append(f"| {rank} | `{c_row.Classifier}` | {c_row.Average_Accuracy:.6f} | {c_row.Files_Count} |")
        report_lines.append("")
        
    # Save detailed report to file
    output_path = os.path.join(directory, "classifier_ranking_report.md")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved detailed report to: {output_path}")

if __name__ == "__main__":
    main()
