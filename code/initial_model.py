import pandas as pd
import numpy as np

# Loading the data
data = {
    "Aggregated_Quarter" : [0, 1, 2, 3, 4, 5, 6, 7],
    "Quarter": ["Q1'26", "Q2'26", "Q3'26", "Q4'26", "Q1'27", "Q2'27", "Q3'27", "Q4'27"],
    "TAM (±2 billion GBs)": [21.8, 27.4, 34.9, 39.0, 44.7, 51.5, 52.5, 53.5],
    "Contribution margin per GB": [0.002] * 8
}

tam_summary = pd.DataFrame(data)
tam_summary

data = {
    "Quarter": ["Q1'26", "Q2'26", "Q3'26", "Q4'26", "Q1'27", "Q2'27", "Q3'27", "Q4'27"],
    "Node 1 - GB per wafer": [100000] * 8,
    "Node 1 - Yield (Percent)": [98, 98, 98, 98, 98, 98, 98, 98],
    "Node 2 - GB per wafer": [150000] * 8,
    "Node 2 - Yield (Percent)": [60, 82, 95, 98, 98, 98, 98, 98],
    "Node 3 - GB per wafer": [270000] * 8,
    "Node 3 - Yield (Percent)": [20, 25, 35, 50, 65, 85, 95, 98]
}

node_yield = pd.DataFrame(data)
node_yield["Node 1 - yield/wafer"] = node_yield["Node 1 - GB per wafer"] * node_yield["Node 1 - Yield (Percent)"] // 100
node_yield["Node 2 - yield/wafer"] = node_yield["Node 2 - GB per wafer"] * node_yield["Node 2 - Yield (Percent)"] // 100
node_yield["Node 3 - yield/wafer"] = node_yield["Node 3 - GB per wafer"] * node_yield["Node 3 - Yield (Percent)"] // 100
node_yield

# Sequential Optimization
from math import floor
quarters = tam_summary["Quarter"].tolist()
TAM_data = {row["Quarter"]: row["TAM (±2 billion GBs)"] * 1e9 for _, row in tam_summary.iterrows()}

initial_loading = {"Node1": 12000, "Node2": 5000, "Node3": 1000}
nodes = ["Node1", "Node2", "Node3"]

sequential_results = {}

for i, q in enumerate(quarters):
    model = LpProblem(f"Maximize_Contribution_{q}", LpMaximize)

    x = {n: LpVariable(f"x_{q}_{n}", lowBound=0) for n in nodes}

    model += lpSum(0.002 * 13 * x[n] * yield_per_wafer[n][q] for n in nodes), "Quarter_Contribution"

    model += 13 * lpSum(x[n] * yield_per_wafer[n][q] for n in nodes) - TAM_data[q] <= 2 * 1e9, f"TAM_upper_bound_{q}"
    model += TAM_data[q] - 13 * lpSum(x[n] * yield_per_wafer[n][q] for n in nodes) <= 2 * 1e9, f"TAM_lower_bound_{q}"
    if i > 0:
        prev_q = quarters[i - 1]
        for n in nodes:
            prev_loading = sequential_results[prev_q][n]
            model += x[n] - prev_loading <= 2500, f"Loading_change_up_{q}_{n}"
            model += prev_loading - x[n] <= 2500, f"Loading_change_down_{q}_{n}"
    else:
        for n in nodes:
            model += x[n] == initial_loading[n], f"Initial_loading_{n}"

    model.solve()

    quarter_solution = {n: floor(x[n].varValue) for n in nodes}
    sequential_results[q] = quarter_solution

    print(f"Status for {q}: {LpStatus[model.status]}")
    print(f"{q} Solution: {quarter_solution}\n")

print("Sequential Optimization Results:")
for q, solution in sequential_results.items():
    print(f"{q}: {solution}")

# Verifying the total output
total_gb_output = {}

for q in quarters:
    if q in sequential_results:  # Ensure the quarter has a valid solution
        gb_output = 13 * sum(
            sequential_results[q][n] * yield_per_wafer[n][q] for n in nodes
        )
        total_gb_output[q] = round(gb_output)

# Print results
print("Total GB Output per Quarter:")
for q, output in total_gb_output.items():
    print(f"{q}: {output} GB")


# Calculate tools needed for workstation
data = {
    "Workstation": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    "Initial tool count for Q1'26": [10, 18, 5, 11, 15, 2, 23, 3, 4, 1],
    "Utilization (percent)": [78, 76, 80, 80, 76, 80, 70, 85, 75, 60],
    "CAPEX per tool (million dollars)": [3.0, 6.0, 2.2, 3.0, 3.5, 6.0, 2.1, 1.8, 3.0, 8.0],
    "Node 1 Minute Load": [4.0, 6.0, 2.0, 5.0, 5.0, 0, 12.0, 2.1, 0, 0],
    "Node 2 Minute Load": [4.0, 9.0, 2.0, 5.0, 10.0, 1.8, 0, 0, 6.0, 0],
    "Node 3 Minute Load": [4.0, 15.0, 5.4, 0, 0, 5.8, 16.0, 0, 0, 2.1]
}

workstation = pd.DataFrame(data)
workstation.fillna(0, inplace = True)
workstation

import math
# Initialize an empty dctionary for total tool requirements
tool_requirements = [] # Tool requirement for each workstation
quarter_tool_requirement = [] # Tool requirement for each quarter
workstation_quarter_tool = [] * 8

# Collect the initial tool count for each workstation for Q1'26
for idx, row in workstation.iterrows():
    tool_requirements.append(row["Initial tool count for Q1'26"])

print(tool_requirements)
prev_tool = tool_requirements[:]
quarter_tool_requirement.append(sum(tool_requirements))
quarters_new = quarters[:]
del quarters_new[0]

for q_idx, q in enumerate(quarters_new):
  work_station = []
  for idx, row in workstation.iterrows():
    total_tool_requirement = 0 # This is the total tool requirement for each workstation
    for n_idx, n in enumerate(nodes): # Iterate over each node
      minute_load = row[f"Node {n_idx+1} Minute Load"]
      utilization = row["Utilization (percent)"] / 100
      loading = int(sequential_results[q][n])
      node_tool_requirement = (loading * minute_load) / (7 * 24 * 60 * utilization)
      total_tool_requirement += math.ceil(node_tool_requirement)
    tool_requirements[idx] = math.ceil(total_tool_requirement) # The new initial tool count for the quarter is the total of the prev
    work_station.append(math.ceil(total_tool_requirement))
  workstation_quarter_tool.append(work_station)
  quarter_tool_requirement.append(sum(tool_requirements))

# Calculate additional CAPEX incurred
cost_quarter = [[0] * 10 for _ in range(8)]  # Initialize CAPEX per quarter

for q in range(8):
    for w in range(len(workstation)):  # Loop through workstations
        capex_per_tool = workstation["CAPEX per tool (million dollars)"][w]
        new_tools = max(workstation_quarter_tool[q][w] - prev_tool[w], 0)
        cost_quarter[q][w] += new_tools * capex_per_tool

        # Update prev_tool for the next quarter
        prev_tool[w] = max(workstation_quarter_tool[q][w], prev_tool[w])
