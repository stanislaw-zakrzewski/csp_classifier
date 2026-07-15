import os
import glob
import pandas as pd
import argparse

def evaluate_classifiers(directory):
    print(f"Reading CSV files from directory: {directory}")
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # Filter out any previously generated output files to avoid reading them
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("averaged_accuracies")]
    
    if not csv_files:
        print(f"No valid CSV files found in {directory}")
        return
        
    all_final_accuracies = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # The final accuracy for a classifier is the Cumulative_Accuracy 
            # at the maximum Trial index for that classifier in this file.
            idx = df.groupby('Classifier')['Trial'].idxmax()
            final_accs = df.loc[idx, ['Classifier', 'Cumulative_Accuracy']].copy()
            final_accs['Source_File'] = os.path.basename(file)
            
            all_final_accuracies.append(final_accs)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    if not all_final_accuracies:
        print("No data extracted from CSV files.")
        return
        
    # Combine all extracted final accuracies
    combined_df = pd.concat(all_final_accuracies, ignore_index=True)
    
    # Calculate the average final accuracy across the 10 best runs for each classifier
    averaged_df = combined_df.groupby('Classifier').agg(
        Average_Accuracy=('Cumulative_Accuracy', lambda x: x.nlargest(10).mean()),
        Files_Count=('Source_File', 'count')
    ).reset_index()
    
    # Sort the classifiers by average accuracy in descending order
    averaged_df = averaged_df.sort_values(by='Average_Accuracy', ascending=False)
    
    # Save to a new CSV file
    output_path = os.path.join(directory, "averaged_accuracies.csv")
    averaged_df.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully evaluated {len(csv_files)} files.")
    print(f"Results sorted by average accuracy and saved to {output_path}\n")
    
    # Display the top 15 results
    print("Top 15 Classifiers:")
    print(averaged_df.head(15).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classifier accuracies across multiple simulation CSV files.")
    parser.add_argument(
        "--directory", 
        type=str, 
        default=os.path.normpath("simulation_results/PhysionetMI"),
        help="Directory containing the simulation CSV files"
    )
    
    args = parser.parse_args()
    
    if os.path.isdir(args.directory):
        evaluate_classifiers(args.directory)
    else:
        print(f"Error: The directory '{args.directory}' does not exist.")
