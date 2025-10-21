############## 4.1 Flu Surveillance using Google Symptoms Data ##############
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr

ilin_state_path = "/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/ILINet_states.csv"
df_state = pd.read_csv(ilin_state_path)
df_ga = df_state[df_state["REGION"] == "Georgia"].copy()

def make_week_date(year, week):
    try:
        return datetime.fromisocalendar(int(year), int(week), 7)
    except ValueError:
        return pd.NaT
df_ga["WEEK_DATE"] = [make_week_date(y, w) for y, w in zip(df_ga["YEAR"], df_ga["WEEK"])]

# Season = week 40 of 2018 through week 20 of 2019
mask_2018 = (df_ga["YEAR"] == 2018) & (df_ga["WEEK"] >= 40)
mask_2019 = (df_ga["YEAR"] == 2019) & (df_ga["WEEK"] <= 20)
df_ga_season = df_ga[mask_2018 | mask_2019].copy()
df_ga_season = df_ga_season.sort_values("WEEK_DATE")
df_ga_season["%UNWEIGHTED ILI"] = pd.to_numeric(df_ga_season["%UNWEIGHTED ILI"], errors="coerce")
print(f"Filtered Georgia ILI data for 2018–2019 flu season: {len(df_ga_season)} weeks")

#plot % unweighted IL
plt.figure(figsize=(10, 5))
plt.plot(df_ga_season["WEEK_DATE"], df_ga_season["%UNWEIGHTED ILI"], marker="o", color="blue")
plt.xlabel("Week (end date)")
plt.ylabel("% Unweighted ILI")
plt.title("Georgia: Weekly % Unweighted ILI (2018–2019 Flu Season)")
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/Scripts/q4/Figures/q4_1.png", dpi=150)
plt.show()

#################################
############## 4.2 ##############
#################################

symp_2018_path = "/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/2018_symptoms_dataset.csv"
symp_2019_path = "/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/2019_symptoms_dataset.csv"

symptom_list = ["symptom:Fever","symptom:Low-grade fever","symptom:Cough","symptom:Sore throat","symptom:Headache","symptom:Fatigue", "symptom:Muscle weakness"]

df18 = pd.read_csv(symp_2018_path)
df19 = pd.read_csv(symp_2019_path)
df_all = pd.concat([df18, df19], ignore_index=True)

df_ga = df_all[df_all["sub_region_1"] == "Georgia"]
df_ga["date"] = pd.to_datetime(df_ga["date"])

df_ga = df_ga[["date"] + symptom_list]
df_ga = df_ga[(df_ga["date"] >= "2018-10-01") & (df_ga["date"] <= "2019-05-31")]

df_ga = df_ga.set_index("date").resample("W-SUN").mean().reset_index()
df_ga["Epiweek"] = df_ga.index + 1

plt.figure(figsize=(12, 6))
for col in symptom_list:
    plt.plot(df_ga["Epiweek"], df_ga[col], label=col.replace("symptom:", ""))

plt.xlabel("Epiweek (2018–2019 Season)")
plt.ylabel("Symptom Trend Value")
plt.title("Google Symptoms Trends for Flu-related Symptoms (Georgia, 2018 – 2019 Season)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/Scripts/q4/Figures/q4_2.png", dpi=150)
plt.show()


#################################
############## 4.3 ##############
#################################
ili = pd.read_csv(ilin_state_path)
ili_ga = ili[ili["REGION"] == "Georgia"]
ili_ga = ili_ga[((ili_ga["YEAR"] == 2018) & (ili_ga["WEEK"] >= 40)) |
                ((ili_ga["YEAR"] == 2019) & (ili_ga["WEEK"] <= 20))]
ili_ga = ili_ga[["YEAR", "WEEK", "%UNWEIGHTED ILI"]].rename(columns={"%UNWEIGHTED ILI": "ILI"})
ili_ga = ili_ga.sort_values(["YEAR", "WEEK"]).reset_index(drop=True)
ili_ga["Epiweek"] = ili_ga.index + 1

# Symptom weekly (df_ga from 4.2 already has weekly means + Epiweek)
df_sym_week = df_ga[["Epiweek"] + symptom_list]

# Merge and compute PCC
merged = pd.merge(ili_ga[["Epiweek", "ILI"]], df_sym_week, on="Epiweek", how="inner")

merged["ILI"] = pd.to_numeric(merged["ILI"], errors="coerce")
print("Pearson Correlation Coefficients:")
for col in symptom_list:
    r, p = pearsonr(merged["ILI"], merged[col])
    print(f"{col.replace('symptom:', '')}: r = {r:.2f}, p = {p:.3g}")