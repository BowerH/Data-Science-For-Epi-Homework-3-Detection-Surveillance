#################################
############## 4.4 ##############
#################################

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr

fig_dir = "/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/Scripts/q4/Figures"
states = ["Georgia", "California", "Texas", "New York", "Alaska", "Mississippi"]

ili_all = pd.read_csv("/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/ILINet_states.csv")
sym18 = pd.read_csv("/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/2018_symptoms_dataset.csv")
sym19 = pd.read_csv("/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/2019_symptoms_dataset.csv")
sym_all = pd.concat([sym18, sym19], ignore_index=True)

symptom_list = ["symptom:Fever","symptom:Low-grade fever","symptom:Cough","symptom:Sore throat","symptom:Headache","symptom:Fatigue", "symptom:Muscle weakness"]

def make_week_date(year, week):
    try:
        return datetime.fromisocalendar(int(year), int(week), 7)  # Sunday
    except Exception:
        return pd.NaT

for state in states:
    ili_s = ili_all[ili_all["REGION"] == state].copy()
    ili_s = ili_s[((ili_s["YEAR"] == 2018) & (ili_s["WEEK"] >= 40)) |
                  ((ili_s["YEAR"] == 2019) & (ili_s["WEEK"] <= 20))]
    ili_s = ili_s.sort_values(["YEAR", "WEEK"]).reset_index(drop=True)
    ili_s["WEEK_DATE"] = [make_week_date(y, w) for y, w in zip(ili_s["YEAR"], ili_s["WEEK"])]
    ili_s["%UNWEIGHTED ILI"] = pd.to_numeric(ili_s["%UNWEIGHTED ILI"], errors="coerce")

    plt.figure(figsize=(10, 5))
    plt.plot(ili_s["WEEK_DATE"], ili_s["%UNWEIGHTED ILI"], marker="o")
    plt.xlabel("Week (end date)")
    plt.ylabel("% Unweighted ILI")
    plt.title(f"{state}: Weekly % Unweighted ILI (2018–2019 Flu Season)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/q4_4_ILI_{state.replace(' ', '_')}.png", dpi=150)
    plt.close()

    sym_s = sym_all[sym_all["sub_region_1"] == state].copy()
    sym_s["date"] = pd.to_datetime(sym_s["date"])
    sym_s = sym_s[["date"] + symptom_list]
    sym_s = sym_s[(sym_s["date"] >= "2018-10-01") & (sym_s["date"] <= "2019-05-31")]
    sym_s = sym_s.set_index("date").resample("W-SUN").mean().reset_index()
    sym_s["Epiweek"] = sym_s.index + 1

    plt.figure(figsize=(12, 6))
    for col in symptom_list:
        plt.plot(sym_s["Epiweek"], sym_s[col], label=col.replace("symptom:", ""))
    plt.xlabel("Epiweek (2018–2019 Season)")
    plt.ylabel("Symptom Trend Value")
    plt.title(f"Google Symptoms Trends for Flu-related Symptoms ({state}, 2018–2019)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/q4_4_symptoms_{state.replace(' ', '_')}.png", dpi=150)
    plt.close()

    ili_ep = ili_s[["%UNWEIGHTED ILI"]].rename(columns={"%UNWEIGHTED ILI": "ILI"}).copy()
    ili_ep["Epiweek"] = range(1, len(ili_ep) + 1)

    merged = pd.merge(ili_ep[["Epiweek", "ILI"]], sym_s[["Epiweek"] + symptom_list],on="Epiweek", how="inner")
    merged["ILI"] = pd.to_numeric(merged["ILI"], errors="coerce")
    best_symptom = None
    best_abs = -1.0
    best_r = None

    for col in symptom_list:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        pair = merged[["ILI", col]].dropna()
        if len(pair) < 2:
            continue
        r, _ = pearsonr(pair["ILI"], pair[col])
        if abs(r) > best_abs:
            best_abs = abs(r)
            best_r = r
            best_symptom = col.replace("symptom:", "")

    print(f"{state}: max PCC -> {best_symptom} (r = {best_r:.3f})")


##################################
############## 4.5 ###############
##################################

T = len(merged)
symptom = "symptom:Fever" 
ili = merged["ILI"].to_numpy()
sym_values = merged[symptom].to_numpy()

def compute_lead_time(ili_series, sym_series, T):
    max_pcc = -2  
    best_t = 0
    for t_prime in range(0, min(10, T - 2)):
        ili_shifted = ili_series[t_prime+1 : T]
        sym_shifted = sym_series[: T - t_prime - 1]
        if len(ili_shifted) < 3:
            continue
        r, _ = pearsonr(ili_shifted, sym_shifted)
        if r > max_pcc:
            max_pcc = r
            best_t = t_prime
    return best_t, max_pcc

lead_time, best_r = compute_lead_time(ili, sym_values, T)
print(f"Lead time for {symptom.replace('symptom:', '')}: {lead_time} weeks (PCC={best_r:.3f})")