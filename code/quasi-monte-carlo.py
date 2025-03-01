import numpy as np
import pandas as pd
import math
from scipy.stats import qmc, gaussian_kde
from math import floor
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus

## With the optimized model
# Table 1: TAM summary
tam_summary = pd.DataFrame({
    "Aggregated_Quarter": [0, 1, 2, 3, 4, 5, 6, 7],
    "Quarter": ["Q1'26", "Q2'26", "Q3'26", "Q4'26", "Q1'27", "Q2'27", "Q3'27", "Q4'27"],
    "TAM (±2 billion GBs)": [21.8, 27.4, 34.9, 39.0, 44.7, 51.5, 52.5, 53.5],
    "Contribution margin per GB": [0.002] * 8
})
# Table 2: Node yield and wafer info
node_yield = pd.DataFrame({
    "Quarter": ["Q1'26", "Q2'26", "Q3'26", "Q4'26", "Q1'27", "Q2'27", "Q3'27", "Q4'27"],
    "Node 1 - GB per wafer": [100000] * 8,
    "Node 1 - Yield (Percent)": [98, 98, 98, 98, 98, 98, 98, 98],
    "Node 2 - GB per wafer": [150000] * 8,
    "Node 2 - Yield (Percent)": [60, 82, 95, 98, 98, 98, 98, 98],
    "Node 3 - GB per wafer": [270000] * 8,
    "Node 3 - Yield (Percent)": [20, 25, 35, 50, 65, 85, 95, 98]
})
# Compute effective yield (GB per wafer) for each node
node_yield["Node 1 yield"] = node_yield["Node 1 - GB per wafer"] * node_yield["Node 1 - Yield (Percent)"] / 100
node_yield["Node 2 yield"] = node_yield["Node 2 - GB per wafer"] * node_yield["Node 2 - Yield (Percent)"] / 100
node_yield["Node 3 yield"] = node_yield["Node 3 - GB per wafer"] * node_yield["Node 3 - Yield (Percent)"] / 100

# Table 3: Workstation data
workstation = pd.DataFrame({
    "Workstation": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "Initial tool count for Q1'26": [10, 18, 5, 11, 15, 2, 23, 3, 4, 1],
    "Utilization (percent)": [78, 76, 80, 80, 76, 80, 70, 85, 75, 60],
    "CAPEX per tool (million dollars)": [3.0, 6.0, 2.2, 3.0, 3.5, 6.0, 2.1, 1.8, 3.0, 8.0],
    "Node 1 Minute Load": [4.0, 6.0, 2.0, 5.0, 5.0, 0, 12.0, 2.131, 0, 0],
    "Node 2 Minute Load": [4.0, 9.0, 2.0, 5.0, 10.0, 1.8, 0, 0, 5.992, 0],
    "Node 3 Minute Load": [4.0, 15.0, 5.4, 0, 0, 5.8, 16.0, 0, 0, 3.024]
})
workstation.fillna(0, inplace=True)

# Define the number of minutes available per tool per quarter.
M = 13 * 10080

quarters = tam_summary["Quarter"].tolist()
nodes = ['Node 1', 'Node 2', 'Node 3']   # Consistent naming!
ws_names = workstation["Workstation"].tolist()

# Mapping: TAM target (in billions of GB) and contribution margin for each quarter
TAM_target = {q: tam for q, tam in zip(tam_summary["Quarter"], tam_summary["TAM (±2 billion GBs)"])}
contrib_margin = {q: cm for q, cm in zip(tam_summary["Quarter"], tam_summary["Contribution margin per GB"])}

# Mapping: effective yield per wafer for each node in each quarter
yield_per_wafer = {}
for node in nodes:
    col = node + " yield"
    yield_per_wafer[node] = {q: node_yield.loc[node_yield["Quarter"] == q, col].values[0] for q in quarters}

# Mapping: Workstation parameters (initial tool count, utilization, CAPEX, minute loads per node)
init_tools = {w: workstation.loc[workstation["Workstation"] == w, "Initial tool count for Q1'26"].values[0] for w in ws_names}
utilization = {w: workstation.loc[workstation["Workstation"] == w, "Utilization (percent)"].values[0] for w in ws_names}
capex = {w: workstation.loc[workstation["Workstation"] == w, "CAPEX per tool (million dollars)"].values[0] for w in ws_names}

# Minute load per node for each workstation
minute_load = {w: {} for w in ws_names}
for w in ws_names:
    for node in nodes:
        col = node + " Minute Load"
        minute_load[w][node] = workstation.loc[workstation["Workstation"] == w, col].values[0]

# Model Formulation using PuLP
model = LpProblem("Maximize_Total_Profit", LpMaximize)

# Decision Variables:
# x[q, node]: number of wafers produced in quarter q for wafer type "node"
x = {(q, node): LpVariable(f"x_{q}_{node}", lowBound=0) for q in quarters for node in nodes}

# delta_y[w,q]: additional tools purchased at workstation w in quarter q (integer)
delta_y = {(w, q): LpVariable(f"delta_{w}_{q}", lowBound=0, cat="Integer") for w in ws_names for q in quarters}

# y[w,q]: cumulative number of tools available at workstation w in quarter q (integer)
y = {(w, q): LpVariable(f"y_{w}_{q}", lowBound=0, cat="Integer") for w in ws_names for q in quarters}

# Objective Function
model += lpSum(
    contrib_margin[q] * lpSum(x[q, node] * yield_per_wafer[node][q] for node in nodes) -
    lpSum(capex[w] * delta_y[w, q] for w in ws_names)
    for q in quarters
), "Total_Profit"

# Constraints
# 1. Production (TAM) Constraint for each quarter:
for q in quarters:
    model += lpSum(13 * x[q, node] * yield_per_wafer[node][q] for node in nodes) <= (TAM_target[q] + 2) * 1e9, f"TAM_upper_{q}"
    model += lpSum(13 * x[q, node] * yield_per_wafer[node][q] for node in nodes) >= (TAM_target[q] - 2) * 1e9, f"TAM_lower_{q}"

# 2. ADDITIONAL Workstation Capacity Constraints for each workstation and quarter:
for q in quarters:
    for w in ws_names:
        model += lpSum(minute_load[w][node] * x[q, node] for node in nodes) <= y[w, q] * 10080 * (utilization[w] / 100), f"Capacity_{w}_{q}"

# 3. Cumulative Capacity Constraints:
first_q = quarters[0]
for w in ws_names:
    model += y[w, first_q] == init_tools[w] + delta_y[w, first_q], f"InitCapacity_{w}_{first_q}"

for idx, q in enumerate(quarters[1:], start=1):
    prev_q = quarters[idx - 1]
    for w in ws_names:
        model += y[w, q] == y[w, prev_q] + delta_y[w, q], f"CumulCapacity_{w}_{q}"

# 4. Loading Constraints:
# Initial loading values for wafer production (number of wafers) for Q1'26
init_loading = {"Node 1": 12000, "Node 2": 5000, "Node 3": 1000}
first_q = quarters[0]
for node in nodes:
    model += x[first_q, node] == init_loading[node], f"Init_loading_{node}"

# For subsequent quarters, the change in production for each node is restricted to ±2500 wafers.
for idx, q in enumerate(quarters[1:], start=1):
    prev_q = quarters[idx - 1]
    for node in nodes:
        model += x[q, node] - x[prev_q, node] <= 2500, f"Loading_change_pos_{node}_{q}"
        model += x[prev_q, node] - x[q, node] <= 2500, f"Loading_change_neg_{node}_{q}"

# Solve the Model
model.solve()

# -----------------------------
# Output the Results
# -----------------------------
print("Status:", LpStatus[model.status])
print("\nOptimal Production Plan (number of wafers):")
for q in quarters:
    prod = {node: floor(x[q, node].varValue) for node in nodes}
    print(f"{q}: {prod}")

print("\nCumulative Tools by Workstation:")
for q in quarters:
    cap = {w: y[w, q].varValue for w in ws_names}
    print(f"{q}: {cap}")

print("\nAdditional Tools Purchased (per quarter):")
for q in quarters:
    invest = {w: delta_y[w, q].varValue for w in ws_names}
    print(f"{q}: {invest}")


file_path = "RPT Data.xlsx"
sheet_name = "Question 2 RPT Data"
df = pd.read_excel(file_path, sheet_name=sheet_name)
df['J_3'] = power_transformer.fit_transform(df_rpt[['J_3']])

# Print the first few rows to verify

# Assume the columns are named "H_1", "I_2", "J_3" for Workstations H, I, J
workstations = {"H": "H_1", "I": "I_2", "J": "J_3"}

num_wafers_per_node = {"H": 12000, "I": 5000, "J": 1000}
num_simulations = 10000  # Number of Monte Carlo trials

results = {}

for station, col_name in workstations.items():
    # Get RPT values for this workstation
    rpt_population = df[col_name].dropna().values
    n_population = len(rpt_population)

    # Generate QMC samples using Sobol' sequence
    sampler = qmc.Sobol(d=1, scramble=True)
    ## qmc_samples = sampler.random(num_simulations * num_wafers_per_node[station])

    num_samples = num_simulations * num_wafers_per_node[station]
    power_of_2_samples = 2 ** math.ceil(math.log2(num_samples))  # Get next power of 2

    sampler = qmc.Sobol(d=1, scramble=True)
    qmc_samples = sampler.random(power_of_2_samples)[:num_samples]  # Trim excess samples

    # Convert QMC samples to indices in the RPT dataset
    indices = (qmc_samples * n_population).astype(int).flatten() % n_population
    sampled_rpts = rpt_population[indices].reshape(num_simulations, num_wafers_per_node[station])

    # Compute total processing time for all wafers in each simulation
    total_times = np.sum(sampled_rpts, axis=1)

    # Compute statistical summaries
    mean_total_time = np.mean(total_times)
    std_total_time = np.std(total_times)
    percentiles = np.percentile(total_times, [5, 50, 95])  # 5th, median, 95th percentile

    # Store results
    results[station] = {
        "Mean": mean_total_time,
        "Std Dev": std_total_time,
        "5th Percentile": percentiles[0],
        "Median": percentiles[1],
        "95th Percentile": percentiles[2]
    }

for station, stats in results.items():
    print(f"Workstation {station}:")
    print(f"  Estimated Mean Total Processing Time: {stats['Mean']:.2f} units")
    print(f"  Standard Deviation: {stats['Std Dev']:.2f}")
    print(f"  5th Percentile: {stats['5th Percentile']:.2f}, Median: {stats['Median']:.2f}, 95th Percentile: {stats['95th Percentile']:.2f}")
    print("-" * 50)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

import pandas as pd

file_path = 'RPT Data.xlsx'
data = pd.ExcelFile(file_path)

# Display sheet names to understand the structure of the file
sheet_names = data.sheet_names
sheet_names

# Load the data
file_path = 'RPT Data.xlsx'
df = pd.read_excel(file_path, sheet_name='Question 2 RPT Data')

print(df.head())


# Columns for processing times are labeled as 'H1', 'I2', and 'J3'
cols = ['H_1', 'I_2', 'J_3']
pop_params = {}
for col in cols:
    pop_params[col] = {'mean': df[col].mean(), 'std': df[col].std()}
    print(col + ' parameters:', pop_params[col])

# Define wafer counts based on simulation input
wafer_counts =  {

  'H_1': [12000, 11925, 9425, 6925, 7236, 8189, 6725, 6849],
  'I_2': [5000, 6993, 9493, 10093, 10062, 10093, 10093, 10093],
  'J_3': [1000, 3446, 5946, 7343, 8000, 7969, 7990, 7990]

}


n_samples = 8192

# Define a function to compute empirical quantile using interpolation
def empirical_quantile(data, u):
    data_sorted = np.sort(data)
    n = len(data_sorted)
    # Create an evenly spaced grid between 0 and 1 for quantiles
    grid = np.linspace(0, 1, n)
    return np.interp(u, grid, data_sorted)

# Function to perform QMC simulation for empirical distribution summing count draws in blocks to avoid memory issues

def qmc_empirical_sum(data, count, n_samples, block_size=100):
    total_samples = np.zeros(n_samples)
    remaining = count
    # Initialize a Sobol engine with dimension equal to block_size
    while remaining > 0:
        current_block = min(block_size, remaining)
        sobol = qmc.Sobol(d=current_block, scramble=True)
        # Generate quasi random numbers in shape (n_samples, current_block)
        U = sobol.random(n_samples)
        # Map these using the empirical quantile function for each column
        # Loop over current block dimension
        block_vals = np.zeros((n_samples, current_block))
        for j in range(current_block):
            block_vals[:, j] = empirical_quantile(data, U[:, j])
        # Sum along the block dimension and add to total
        total_samples += block_vals.sum(axis=1)
        remaining -= current_block
    return total_samples

# Use the function for each node and each week's wafer count
qmc_simulation = {}
for node, weekly_counts in wafer_counts.items():
    data = df[node].dropna().values  # in case there are NaNs
    qmc_simulation[node] = []  # Store results per week
    for week_idx, count in enumerate(weekly_counts):
        qmc_total = qmc_empirical_sum(data, count, n_samples)
        qmc_simulation[node].append(qmc_total)
        print(f'QMC simulated total time for {node}, Quarter {week_idx+1} (first 5 samples):', qmc_total[:5])

# Plot histograms for each node's QMC simulation
plt.figure(figsize=(10, 6))
for node in wafer_counts.keys():
    for week_idx, week_data in enumerate(qmc_simulation[node]):
        plt.hist(np.log(week_data), bins=50, alpha=0.5, label=f'{node} Quarter {week_idx+1}')
plt.title('Histogram of Total Processing Times (QMC Empirical Simulation)')
plt.xlabel('Total Processing Time')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print('QMC Empirical simulation complete')

import numpy as np
list_h1 = [3] * 8
list_i2 = [4,8,8,8,8,8,8,8]
list_j3 = [1,4,4,4,4,4,4,4]
# Define available tool times for each node
available = {'H_1': [x * 8568 for x in list_h1], 'I_2': [x * 7560 for x in list_i2], 'J_3': [x * 6048 for x in list_j3]}

# Calculate probability that total processing time > available tool time for each node and each quarter
probabilities_per_quarter = {}

for node in available.keys():
    probabilities_per_quarter[node] = []
    for quarter_idx in range(len(available[node])):
        sim_data = qmc_simulation[node][quarter_idx]  # Get simulated data for this quarter
        prob = np.mean(sim_data > available[node][quarter_idx])  # Compute probability
        probabilities_per_quarter[node].append(prob)
        print(f'Probability that total time exceeds tool time for {node}, Quarter {quarter_idx+1}:', prob)

# Compute overall failure probability per quarter
overall_probabilities_per_quarter = []

for quarter_idx in range(len(available['H_1'])):  # All lists are of same length
    meet_probs = [1 - probabilities_per_quarter[node][quarter_idx] for node in available.keys()]
    overall_failure_prob = 1 - (np.prod(meet_probs))**13  # P(fail overall) = 1 - P(all nodes meet time constraints)
    overall_probabilities_per_quarter.append(overall_failure_prob)
    print(f'Overall probability that design output is not met, Quarter {quarter_idx+1}:', overall_failure_prob)

print('Done computing probabilities per quarter.')
