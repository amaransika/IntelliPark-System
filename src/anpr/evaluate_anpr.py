import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

csv_path = '../results/parking_log.csv'
output_folder = 'results/charts/'
os.makedirs(output_folder, exist_ok=True)

def generate_anpr_report():
    print("--- Generating IntelliPark Final ANPR Report ---")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run the pipeline first!")
        return
        
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    sns.histplot(df['Confidence'], bins=20, kde=True, color='teal')
    plt.title('Detection Confidence Distribution (IntelliPark Phase 1)')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_folder}confidence_dist.png')
    plt.close()

    status_counts = df['Security_Status'].value_counts()
    
    plt.figure(figsize=(8, 6))
    status_counts.plot(kind='bar', color=['#4CAF50', '#F44336'])
    plt.title('Vehicle Security Status Overview')
    plt.xlabel('Status')
    plt.ylabel('Number of Vehicles')
    plt.xticks(rotation=0)

    for i, v in enumerate(status_counts):
        plt.text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    plt.savefig(f'{output_folder}security_status_report.png')
    plt.close()

    print(f"Success! Charts saved to: {output_folder}")

if __name__ == "__main__":
    generate_anpr_report()