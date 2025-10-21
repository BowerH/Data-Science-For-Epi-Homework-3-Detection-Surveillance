import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

facebook_path = r"/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/facebook.txt"
rand_nodes_path = r"/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/hw3_dataset/datasets/rand_nodes.npy"


####### For question 3.5 replace k = 50  and also k = 500


beta = 0.005          # infection probability
T = 100               # number of time steps
num_runs = 20         # number of independent simulations for averaging
k = 500             # number of sensors for all strategies
seed_size = 4         # 4 nodes infected at t = 0

random = np.random.default_rng()

G = nx.Graph()
with open(facebook_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        u, v = int(parts[0]), int(parts[1])
        G.add_edge(u, v)

all_nodes_list = sorted(G.nodes())
N = len(all_nodes_list)

#random sensors
random_nodes_file = np.load(rand_nodes_path)
random_nodes_in_graph = [int(x) for x in random_nodes_file if int(x) in G]

RANDOM_sensors = set(random_nodes_in_graph[:k])
if len(RANDOM_sensors) < k:
    needed = k - len(RANDOM_sensors)
    candidates = [n for n in all_nodes_list if n not in RANDOM_sensors]
    extra = random.choice(candidates, size=needed, replace=False)
    RANDOM_sensors.update(extra.tolist())
RANDOM_sensors = sorted(RANDOM_sensors)

#friend sensors
FRIENDS_sensors_set = set()
for u in RANDOM_sensors:
    nbrs = list(G.neighbors(u))
    if len(nbrs) == 0:
        v = int(random.choice(all_nodes_list))
    else:
        v = int(random.choice(nbrs))
    FRIENDS_sensors_set.add(v)

while len(FRIENDS_sensors_set) < k:
    u = int(random.choice(RANDOM_sensors))
    nbrs = list(G.neighbors(u))
    v = int(random.choice(nbrs))
    FRIENDS_sensors_set.add(v)

FRIENDS_sensors = sorted(FRIENDS_sensors_set)

# cenytral sensors
centrality = nx.eigenvector_centrality_numpy(G)
CENTRAL_sensors = sorted(centrality.keys(), key=lambda n: centrality[n], reverse=True)[:k]

sum_random_frac = np.zeros(T + 1, dtype=float)
sum_friends_frac = np.zeros(T + 1, dtype=float)
sum_central_frac = np.zeros(T + 1, dtype=float)

#running sims
for run_index in range(num_runs):
    print(f"Run {run_index + 1}/{num_runs}")
    initial_infected_nodes = set(random.choice(all_nodes_list, size=seed_size, replace=False))
    infected_nodes = set(initial_infected_nodes)
    random_frac_ot = np.zeros(T + 1, dtype=float)
    random_frac_ot[0] = len(infected_nodes.intersection(RANDOM_sensors)) / k

    for t in range(1, T + 1):
        new_infections = set()
        for node in all_nodes_list:
            if node in infected_nodes:
                continue
            infected_neighbor_count = 0
            for nbr in G.neighbors(node):
                if nbr in infected_nodes:
                    infected_neighbor_count += 1
            if infected_neighbor_count > 0:
                p_inf = 1.0 - (1.0 - beta) ** infected_neighbor_count
                if random.random() < p_inf:
                    new_infections.add(node)
        infected_nodes |= new_infections
        random_frac_ot[t] = len(infected_nodes.intersection(RANDOM_sensors)) / k
    sum_random_frac += random_frac_ot

    # FRIENDS sensors
    initial_infected_nodes = set(random.choice(all_nodes_list, size=seed_size, replace=False))
    infected_nodes = set(initial_infected_nodes)
    friends_frac_ot = np.zeros(T + 1, dtype=float)
    friends_frac_ot[0] = len(infected_nodes.intersection(FRIENDS_sensors)) / k

    for t in range(1, T + 1):
        new_infections = set()
        for node in all_nodes_list:
            if node in infected_nodes:
                continue
            infected_neighbor_count = 0
            for nbr in G.neighbors(node):
                if nbr in infected_nodes:
                    infected_neighbor_count += 1
            if infected_neighbor_count > 0:
                p_inf = 1.0 - (1.0 - beta) ** infected_neighbor_count
                if random.random() < p_inf:
                    new_infections.add(node)
        infected_nodes |= new_infections
        friends_frac_ot[t] = len(infected_nodes.intersection(FRIENDS_sensors)) / k

    sum_friends_frac += friends_frac_ot

    #CENTRAL sensors
    initial_i_nodes = set(random.choice(all_nodes_list, size=seed_size, replace=False))
    infected_nodes = set(initial_i_nodes)
    central_frac_ot = np.zeros(T + 1, dtype=float)
    central_frac_ot[0] = len(infected_nodes.intersection(CENTRAL_sensors)) / k

    for t in range(1, T + 1):
        new_infections = set()
        for node in all_nodes_list:
            if node in infected_nodes:
                continue
            i_neighbor_count = 0
            for nbr in G.neighbors(node):
                if nbr in infected_nodes:
                    i_neighbor_count += 1
            if i_neighbor_count > 0:
                p_inf = 1.0 - (1.0 - beta) ** i_neighbor_count
                if random.random() < p_inf:
                    new_infections.add(node)
        infected_nodes |= new_infections
        central_frac_ot[t] = len(infected_nodes.intersection(CENTRAL_sensors)) / k
    sum_central_frac += central_frac_ot

#all averages
avg_random_frac = sum_random_frac / num_runs
avg_friends_frac = sum_friends_frac / num_runs
avg_central_frac = sum_central_frac / num_runs

avg_random_daily = np.zeros_like(avg_random_frac)
avg_random_daily[1:] = avg_random_frac[1:] - avg_random_frac[:-1]
avg_random_daily[0] = avg_random_frac[0]

avg_friends_daily = np.zeros_like(avg_friends_frac)
avg_friends_daily[1:] = avg_friends_frac[1:] - avg_friends_frac[:-1]
avg_friends_daily[0] = avg_friends_frac[0]

avg_central_daily = np.zeros_like(avg_central_frac)
avg_central_daily[1:] = avg_central_frac[1:] - avg_central_frac[:-1]
avg_central_daily[0] = avg_central_frac[0]

#plots
#avg fraction infected among sensors vs t
plt.figure(figsize=(10, 5))
plt.plot(range(T + 1), avg_random_frac, label="RANDOM")
plt.plot(range(T + 1), avg_friends_frac, label="FRIENDS")
plt.plot(range(T + 1), avg_central_frac, label="CENTRAL")
plt.xlabel("Time (t)")
plt.ylabel("Average fraction of sensors infected")
plt.title(f"Sensors: Average I(t) vs t (beta={beta}, k={k}, runs={num_runs})")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/Scripts/q3/Figures/q3_5_sensors_avg_fraction_k500.png", dpi=150)

#avg daily change among sensors vs t
plt.figure(figsize=(10, 5))
plt.plot(range(T + 1), avg_random_daily, label="RANDOM")
plt.plot(range(T + 1), avg_friends_daily, label="FRIENDS")
plt.plot(range(T + 1), avg_central_daily, label="CENTRAL")
plt.xlabel("Time (t)")
plt.ylabel("Average daily change among sensors (I_d)")
plt.title(f"Sensors: Average number of daily infections vs t (beta={beta}, k={k}, runs={num_runs})")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/hannah/Tech/DSE/Data-Science-For-Epi-Homework-3-Detection-Surveillance/HW3_export/Scripts/q3/Figures/q3_5_sensors_avg_daily_k500.png", dpi=150)

####################### q3.3 ##########################

t_star = 33 

t_random_peak = int(np.argmax(avg_random_daily))
t_friends_peak = int(np.argmax(avg_friends_daily))
t_central_peak = int(np.argmax(avg_central_daily))

peak_random_value = avg_random_daily[t_random_peak]
peak_friends_value = avg_friends_daily[t_friends_peak]
peak_central_value = avg_central_daily[t_central_peak]

lead_random = t_star - t_random_peak
lead_friends = t_star - t_friends_peak
lead_central = t_star - t_central_peak

print("Sensor Peak and Lead Time")
print(f"RANDOM: peak time = {t_random_peak}, peak daily infection = {peak_random_value:.4f}, lead time = {lead_random}")
print(f"FRIENDS: peak time = {t_friends_peak}, peak daily infection = {peak_friends_value:.4f}, lead time = {lead_friends}")
print(f"CENTRAL: peak time = {t_central_peak}, peak daily infection = {peak_central_value:.4f}, lead time = {lead_central}")
