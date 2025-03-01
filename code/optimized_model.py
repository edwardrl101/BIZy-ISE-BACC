import pandas as pd
from math import floor
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus


# Data Preparation
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

TAM_target = {q: tam for q, tam in zip(tam_summary["Quarter"], tam_summary["TAM (±2 billion GBs)"])}
contrib_margin = {q: cm for q, cm in zip(tam_summary["Quarter"], tam_summary["Contribution margin per GB"])}

yield_per_wafer = {}
for node in nodes:
    col = node + " yield"
    yield_per_wafer[node] = {q: node_yield.loc[node_yield["Quarter"] == q, col].values[0] for q in quarters}

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

# Output the Results
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

# Calculate the profit
additional_tools = {}
additional_capex = 0
for station, num_tools in init_tools.items():
  additional_tools[station] = cap[station] - num_tools
  additional_capex += additional_tools[station] * capex[station]

profit = sum(x for x in total_gb_output.values()) * 0.002 - additional_capex * 1e6
profit / 1e6
