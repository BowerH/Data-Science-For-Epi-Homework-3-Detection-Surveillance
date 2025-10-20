
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Import the SI simulation from your provided sis_model.py
from sis_model import simulate_t_steps_SI


# -----------------------------
# 1) Load the Facebook graph
# -----------------------------
facebook_path = r"C:\Users\hbower\Downloads\HW3_export\hw3_dataset\datasets\facebook.txt"

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


# -----------------------------
# 2) Define SI model parameters
# -----------------------------
beta = 0.005      # infection probability per contact
T = 100           # number of time steps
num_runs = 100    # number of independent simulations
i_frac = 4 / N    # 4 initial infections (fraction of total population)


# -----------------------------
# 3) Prepare data structures
# -----------------------------
# Arrays to store cumulative results for averaging later
sum_S = np.zeros(T + 1, dtype=float)        # total S(t) over all runs
sum_I = np.zeros(T + 1, dtype=float)        # total I(t) over all runs
sum_daily_new = np.zeros(T + 1, dtype=float)  # total new infections per day over all runs


# -----------------------------
# 4) Random Number Generator
# -----------------------------
# Create a random number generator using system entropy
# This ensures every script run is different (not fixed like a base_seed)
rng = np.random.default_rng()

# -----------------------------
# 5) Run the SI simulations
# -----------------------------
for run in range(num_runs):

    # Draw a completely new random integer seed for this run (0 to 1 billion)
    seed = rng.integers(0, 1_000_000_000)
    print(f"Run {run+1} using random seed {seed}")

    # Run one simulation with this random seed
    result = simulate_t_steps_SI(
        G=G,
        i_frac=i_frac,
        beta=beta,
        num_rounds=T,
        seed=seed,
        full_output=False
    )

    # The function returns the counts of S and I at each time step
    S = result["S"].astype(float)
    I = result["I"].astype(float)

    # Compute daily new infections (difference between consecutive I values)
    daily_new = np.zeros_like(I)
    daily_new[1:] = I[1:] - I[:-1]
    daily_new[0] = 0.0   # don’t count initial seeds as “new infections”

    # Accumulate totals for averaging
    sum_S += S
    sum_I += I
    sum_daily_new += daily_new


# -----------------------------
# 6) Compute averages
# -----------------------------
avg_S = sum_S / num_runs
avg_I = sum_I / num_runs
avg_daily_new = sum_daily_new / num_runs

# Convert counts to fractions of population
avg_frac_S = avg_S / N
avg_frac_I = avg_I / N


# -----------------------------
# 7) Find peak day t*
# -----------------------------
t_star = int(np.argmax(avg_daily_new))  # earliest time where new infections peak
print(f"\nAverage peak day t* (maximum average daily new infections): {t_star}")


# -----------------------------
# 8) Plot average fraction S/I vs time
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(T + 1), avg_frac_S, label="Average fraction S(t)")
plt.plot(range(T + 1), avg_frac_I, label="Average fraction I(t)")
plt.xlabel("Time (t)")
plt.ylabel("Fraction of population")
plt.title(f"SI Model on Facebook Network (β={beta}) — Average S and I over {num_runs} runs")
plt.legend()
plt.tight_layout()
plt.savefig("q3_avg_fraction_SI.png", dpi=150)


# -----------------------------
# 9) Plot average daily new infections vs time
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(T + 1), avg_daily_new, color="red", label="Average daily new infections")
plt.xlabel("Time (t)")
plt.ylabel("Average number of new infections")
plt.title(f"SI Model on Facebook Network (β={beta}) — Average Daily New Infections")
plt.legend()
plt.tight_layout()
plt.savefig("q3_avg_daily_new.png", dpi=150)

# Uncomment to view interactively
# plt.show()