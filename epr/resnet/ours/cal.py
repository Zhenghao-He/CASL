import pandas as pd

# Read the CSV file
csv_file = '/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/epr/resnet/ours/epr_results_blond_hair_sae_top1_detailed.csv'
df = pd.read_csv(csv_file)

# Get the last column
last_column = df.iloc[:, -1]

# Define threshold
threshold = 1.1  # Change this value as needed

# Count values greater than threshold
count_above_threshold = (last_column > threshold).sum()
total_count = len(last_column)
percentage = (count_above_threshold / total_count) * 100

print(f"Threshold: {threshold}")
print(f"Values above threshold: {count_above_threshold}")
print(f"Total values: {total_count}")
print(f"Percentage: {percentage:.2f}%")