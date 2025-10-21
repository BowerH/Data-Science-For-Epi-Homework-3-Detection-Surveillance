
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from HW3_export.Scripts.sis_model import simulate_t_steps_SI

facebook_path = r"/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/facebook.txt"

G = nx.Graph()
with open(facebook_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        u_v = line.split()
        if len(u_v) != 2:
            continue
        u, v = int(u_v[0]), int(u_v[1])
        G.add_edge(u, v)

N = G.number_of_nodes()
E = G.number_of_edges()
print(f"Loaded Facebook graph with {N} nodes and {E} edges.")



beta = 0.005      #infection probability per contact
T = 100           #number of time steps
num_runs = 100    #number of independent simulations
i_frac = 4 / N    #4 initial infections

sum_S = np.zeros(T + 1, dtype=float)
sum_I = np.zeros(T + 1, dtype=float)
sum_daily_new = np.zeros(T + 1, dtype=float)

random = np.random.default_rng()

#running sis
for run in range(num_runs):
    seed = random.integers(0, 1_000_000_000)
    print(f"Run {run+1} using random seed {seed}")
    result = simulate_t_steps_SI(G=G,i_frac=i_frac, beta=beta, num_rounds=T, seed=seed, full_output=False)
    S = result["S"].astype(float)
    I = result["I"].astype(float)
    daily_new = np.zeros_like(I)
    daily_new[1:] = I[1:] - I[:-1]
    daily_new[0] = 0.0  
    sum_S += S
    sum_I += I
    sum_daily_new += daily_new


avg_S = sum_S / num_runs
avg_I = sum_I / num_runs
avg_daily_new = sum_daily_new / num_runs

# Convert counts to fractions of population
avg_frac_S = avg_S / N
avg_frac_I = avg_I / N

#peak day t* (maximum average daily new infections)
t_star = int(np.argmax(avg_daily_new)) 
print(f"\nAverage peak day t* : {t_star}")

#plot avaerage fraction S/I vs time
plt.figure(figsize=(10, 5))
plt.plot(range(T + 1), avg_frac_S, label="Average fraction S(t)")
plt.plot(range(T + 1), avg_frac_I, label="Average fraction I(t)")
plt.xlabel("Time (t)")
plt.ylabel("Fraction of population")
plt.title(f"SI Model on Facebook Network (β={beta}): Average over {num_runs} runs")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/q3.1_avg_fraction_SI.png", dpi=150)

#Plot average daily new infections vs t
plt.figure(figsize=(10, 5))
plt.plot(range(T + 1), avg_daily_new, color="red", label="Average daily new infections")
plt.xlabel("Time (t)")
plt.ylabel("Average number of new infections")
plt.title(f"SI Model on Facebook Network (β={beta}) — Average Daily New Infections")
plt.legend()
plt.tight_layout()
plt.savefig("Figures/q3_avg_daily_new.png", dpi=150)
