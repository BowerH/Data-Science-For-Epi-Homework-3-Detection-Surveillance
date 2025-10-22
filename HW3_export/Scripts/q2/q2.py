import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


##### Data cleaning / loading
# Load raw CSV
df = pd.read_csv("/Users/hannah/PycharmProjects/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/nyc_influenza_by_county.csv")
#only INFLUENZA_A
df_A = df[df['Disease'] == 'INFLUENZA_A'].copy()
df_A = df_A[['Week Ending Date', 'County', 'Count']]

# Convert week ending date to datetime
df_A['Week Ending Date'] = pd.to_datetime(df_A['Week Ending Date'])

############# 2.1
county_totals = df_A.groupby('County')['Count'].sum()
top_5_counties = county_totals.nlargest(5).index.tolist()
df_top5 = df_A[df_A['County'].isin(top_5_counties)]
plt.figure(figsize=(12, 6))
for county in top_5_counties:
    county_data = df_top5[df_top5['County'] == county].sort_values('Week Ending Date')
    plt.plot(county_data['Week Ending Date'], county_data['Count'], label=county)
plt.title('Weekly INFLUENZA_A Infections for The Top 5 Counties')
plt.xlabel('Week Ending Date')
plt.ylabel('Weekly Count')
plt.legend()
plt.tight_layout()
#plt.savefig("/Scripts/q2/Figures/top5_influenzaA.png")
plt.show()

############## 2.2

training_df = df_A[df_A['Week Ending Date'] < '2017-01-01']
means = training_df.groupby('County')['Count'].mean()
stds = training_df.groupby('County')['Count'].std()
sorted_counties = sorted(means.index)
mean_str = "Means: " + " ".join([f"{county}: {means[county]:.2f}" for county in sorted_counties])
std_str  = "Stds: "  + " ".join([f"{county}: {stds[county]:.2f}"  for county in sorted_counties])

print()
print("2.2 List of means and standard deviations for each county:")
print(mean_str)
print(std_str)


############## 2.3
test_data = df_A[df_A['Week Ending Date'] >= '2017-01-01'].copy()
def is_anomaly(row):
    county = row['County']
    if county in means and not np.isnan(stds[county]):
        return row['Count'] > means[county] + 3 * stds[county]
    else:
        return False

test_data['is_anomaly'] = test_data.apply(is_anomaly, axis=1)
anomaly_counts = test_data.groupby('County')['is_anomaly'].sum()
sorted_anomaly_counts = anomaly_counts.sort_values(ascending=False)
sorted_anomaly_counts = sorted_anomaly_counts.sort_index().sort_values(ascending=False, kind='mergesort')
top_10 = sorted_anomaly_counts.head(10)
top10out = " ".join([f"{county}: {int(top_10[county])}" for county in top_10.index])
print()
print("2.3 Top 10 counties with most anomalies (over or under) from 2017 and on:")
print(top10out)

############## 2.4  Spatial Subset Scan (with neighbor-averaged counts)

adjacency_path = r"/Users/hannah/PycharmProjects/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/county_adjacency2025.txt"
neighbors = {}

with open(adjacency_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue

        county_full, county_geoid, neighbor_full, neighbor_geoid, _ = parts

        if (", NY" in county_full) and (", NY" in neighbor_full):
            county_clean = county_full.replace(" County", "").replace(", NY", "").strip().upper()
            neighbor_clean = neighbor_full.replace(" County", "").replace(", NY", "").strip().upper()

            if county_clean not in neighbors:
                neighbors[county_clean] = set()
            neighbors[county_clean].add(neighbor_clean)

for county_name in neighbors:
    neighbors[county_name].add(county_name)

df_A['County_data_cleaned'] = df_A['County'].str.upper().str.strip()
df_A['Week Ending Date'] = pd.to_datetime(df_A['Week Ending Date'])

county_neighbor_pairs = []
all_counties_in_data = set(df_A['County_data_cleaned'].unique())
for county_name in all_counties_in_data:
    neighbor_list = neighbors.get(county_name, set([county_name]))
    for neighbor in neighbor_list:
        if neighbor in all_counties_in_data:
            county_neighbor_pairs.append((county_name, neighbor))

neighbor = pd.DataFrame(county_neighbor_pairs, columns=['County_data_cleaned', 'Neighbor_CLEAN'])
neighbor_cnt = df_A[['Week Ending Date', 'County_data_cleaned', 'Count']].copy()
neighbor_cnt = neighbor_cnt.rename(columns={'County_data_cleaned': 'Neighbor_CLEAN', 'Count': 'Neighbor_Count'})
merged_neighbors = neighbor.merge(neighbor_cnt, on='Neighbor_CLEAN', how='left')
average_neighbor_cnt = (merged_neighbors.groupby(['County_data_cleaned', 'Week Ending Date'], as_index=False)['Neighbor_Count'].mean().rename(columns={'Neighbor_Count': 'Neighbor_Avg'}))
df_with_neighbors = df_A.merge(average_neighbor_cnt, on=['County_data_cleaned', 'Week Ending Date'], how='left')
training_data = df_with_neighbors[df_with_neighbors['Week Ending Date'] < '2017-01-01']
testing_data = df_with_neighbors[df_with_neighbors['Week Ending Date'] >= '2017-01-01'].copy()
mean_neighbor = training_data.groupby('County')['Neighbor_Avg'].mean()
std_neighbor = training_data.groupby('County')['Neighbor_Avg'].std()
is_anomaly = []

for index, row in testing_data.iterrows():
    county_name = row['County']
    avg_value = row['Neighbor_Avg']

    if county_name in mean_neighbor and not np.isnan(std_neighbor[county_name]):
        upper_threshold = mean_neighbor[county_name] + 3 * std_neighbor[county_name]
        if avg_value > upper_threshold:
            is_anomaly.append(True)
        else:
            is_anomaly.append(False)
    else:
        is_anomaly.append(False)
testing_data['is_anomaly'] = is_anomaly
anomaly_cnts_per_county = testing_data.groupby('County')['is_anomaly'].sum()
anomaly_cnts_per_county = anomaly_cnts_per_county.sort_index()
anomaly_cnts_per_county = anomaly_cnts_per_county.sort_values(ascending=False)
s_top10 = anomaly_cnts_per_county.head(10)
output_lines = []
for county_name, anomaly_count in s_top10.items():
    output_lines.append(f"{county_name}: {int(anomaly_count)}")
formatted_output = " ".join(output_lines)

print()
print("2.4 Top 10 counties with most anomalies with spatial neighbor averages:")
print(formatted_output)
