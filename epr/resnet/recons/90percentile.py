# import pandas as pd
# import numpy as np

# # Read the CSV file
# # csv_file = '/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/epr/resnet/recons/epr_results_blond_hair_recons_detailed.csv' # 2.32163536
# # csv_file = '/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/epr/resnet/recons/epr_results_big_nose_recons_detailed.csv' # 1.78454723
# csv_file = '/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/epr/resnet/recons/epr_results_beards_recons_detailed.csv' # 2.6291389800000005
# df = pd.read_csv(csv_file)

# # Get the last column
# last_column = df.iloc[:, -1]

# # Calculate the 90th percentile
# percentile_90 = np.percentile(last_column, 90)

# print(f"90th Percentile of '{last_column.name}': {percentile_90}")

import pandas as pd
import numpy as np

# Read the CSV file
# csv_file = '/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/epr/resnet/recons/epr_results_blond_hair_recons_detailed.csv' # 2.32163536
# csv_file = '/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/epr/resnet/ours/epr_results_big_nose_sae_top1_detailed.csv'
csv_file = '/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/epr/resnet/ours/epr_results_blond_hair_sae_top1_detailed.csv'

# csv_file = '/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/epr/resnet/recons/epr_results_big_nose_recons_detailed.csv' # 1.78454723
# csv_file = '/p/realai/zhenghao/LoRA_SAE/Asyrp_SAE/epr/resnet/recons/epr_results_beards_recons_detailed.csv' # 2.6291389800000005
df = pd.read_csv(csv_file)

# Get the last column
last_column = df.iloc[:, -1]

# Calculate statistics
percentile_90 = np.percentile(last_column, 90)
mean = np.mean(last_column)
median = np.median(last_column)
variance = np.var(last_column)
std = np.std(last_column)
mean_plus_2std = mean + 2 * std

print(f"Column name: '{last_column.name}'")
print(f"90th Percentile: {percentile_90}")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Variance: {variance}")
print(f"Standard Deviation: {std}")
print(f"Mean + 2*Std: {mean_plus_2std}")