"""
TUTORIAL — equipment allocation optimization
=============================================================
Projects A, B, C.  Equipment types D (2 units), E (5), F (6).
"""
import pandas as pd
import pulp

# ============================================================
# STEP 1 — LOAD THE DATA
# ============================================================
demand = pd.read_excel("Tutorial_Allocation.xlsx", sheet_name="Demand")
fleet = pd.read_excel("Tutorial_Allocation.xlsx", sheet_name="Fleet")
projects = pd.read_excel("Tutorial_Allocation.xlsx", sheet_name="Projects").set_index("Project")
equipment = pd.read_excel("Tutorial_Allocation.xlsx", sheet_name="Equipment").set_index("Equipment")

print("=" * 60)
print("STEP 1: the problem")
print("=" * 60)
print(f"Demand by type : {demand.groupby('Equipment')['Qty'].sum().to_dict()}")
print(f"Supply by type : {fleet.groupby('Equipment').size().to_dict()}")
print("-> We are SHORT: D by 2, E by 1, F by 2. Someone won't get")
print("   fleet units. The optimizer decides WHO, using the criteria.")

# attach criteria scores to each demand line
demand["Criticality"] = demand["Project"].map(projects["Criticality (1-5)"])
demand["Maint_Team"] = demand["Project"].map(projects["Maintenance_Team (Y/N)"]).eq("Y")
demand["Rentability"] = demand["Equipment"].map(equipment["Rentability (1=easy,5=hard)"])

# ============================================================
# STEP 2 — THE WEIGHTS (tuning knobs)
# ============================================================
# Read from the 'Weights' sheet so you can tune without touching code.
w = pd.read_excel("Tutorial_Allocation.xlsx", sheet_name="Weights").set_index("Parameter")["Value"]

W_SHORTAGE = w["WSHORTAGE"]   # base pain of not covering 1 unit of demand
W_CRIT = w["WCRIT"]           # extra pain per criticality point
W_RENT = w["WRENT"]           # extra pain per rentability point
W_MOVE = w["WMOVE"]           # cost of transferring a unit between projects
W_MOVE_WS = w["WMOVE_WS"]     # cheaper to deploy from Workshop
W_MAINT = w["WMAINT"]         # penalty: owned unit at project with no maintenance team
W_DUR = w["WDUR"]             # bonus per month: owned units prefer long jobs

print(f"\nSTEP 2: weights loaded from 'Weights' sheet: {dict(w)}")

# ============================================================
# STEP 3 — DECISION VARIABLES
# ============================================================
# x[unit, demand_line] = 1  ->  send this unit to that project
# short[demand_line]   = n  ->  n units uncovered (rent them)
pairs = [(u, d) for u in fleet.index for d in demand.index
         if fleet.at[u, "Equipment"] == demand.at[d, "Equipment"]]  # same type only!

prob = pulp.LpProblem("tutorial", pulp.LpMinimize)
x = pulp.LpVariable.dicts("x", pairs, cat="Binary")
short = pulp.LpVariable.dicts("short", demand.index.tolist(), lowBound=0, cat="Integer")

print(f"\nSTEP 3: {len(pairs)} possible unit->project assignments "
      f"(only matching equipment types), {len(demand)} shortage variables.")

# ============================================================
# STEP 4 — COSTS:
# ============================================================
def assign_cost(u, d):
    """Cost of sending unit u to demand line d. Lower = better."""
    cost = 0.0
    if fleet.at[u, "Current_Location"] != demand.at[d, "Project"]:      # criterion: minimize transfers
        cost += W_MOVE_WS if fleet.at[u, "Current_Location"] == "Workshop" else W_MOVE
    if not demand.at[d, "Maint_Team"]:                                  # criterion 1: maintenance
        cost += W_MAINT          # owned unit with no team to maintain it -> discouraged
    cost -= W_DUR * demand.at[d, "Duration_Months"]                     # criterion 2: duration
    return cost                  # long job -> bigger bonus -> owned unit preferred there

def shortage_cost(d):
    """Cost of leaving 1 unit of demand d uncovered (= rent it)."""
    return (W_SHORTAGE
            + W_CRIT * demand.at[d, "Criticality"]                      # criterion 3: criticality
            + W_RENT * demand.at[d, "Rentability"])                     # criterion 4: rental ease

print("\nSTEP 4: example shortage costs (what the solver avoids most):")
for d in demand.index:
    r = demand.loc[d]
    print(f"  {r['Project']}-{r['Equipment']}: {shortage_cost(d):>4.0f}"
          f"   (crit={r['Criticality']}, rentability={r['Rentability']})")
print("-> Leaving critical-A short of hard-to-rent D costs 35;")
print("   leaving C short of easy-to-rent F costs only 18.")
print("   So gaps get pushed toward easy-rent types at low-criticality projects.")

# ============================================================
# STEP 5 — OBJECTIVE + CONSTRAINTS, THEN SOLVE
# ============================================================
prob += (pulp.lpSum(x[p] * assign_cost(p) for p in pairs)
         + pulp.lpSum(short[d] * shortage_cost(d) for d in demand.index))

for d in demand.index:   # every demand line fully accounted for: fleet + rent = qty
    prob += pulp.lpSum(x[(u, dd)] for (u, dd) in pairs if dd == d) + short[d] == demand.at[d, "Qty"]

for u in fleet.index:    # each physical unit used at most once
    prob += pulp.lpSum(x[(uu, d)] for (uu, d) in pairs if uu == u) <= 1

prob.solve(pulp.PULP_CBC_CMD(msg=0))

print(f"\nSTEP 5: solver status = {pulp.LpStatus[prob.status]}, "
      f"total cost = {pulp.value(prob.objective):.1f}")

# ============================================================
# STEP 6 — READ THE ANSWER
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: THE ALLOCATION")
print("=" * 60)

rows = []
for (u, d), var in x.items():
    if var.value() > 0.5:
        frm, to = fleet.at[u, "Current_Location"], demand.at[d, "Project"]
        rows.append((fleet.at[u, "Code"], fleet.at[u, "Equipment"], frm, to,
                     "STAY" if frm == to else "TRANSFER"))

alloc = pd.DataFrame(rows, columns=["Code", "Type", "From", "To", "Action"]).sort_values(["Type", "To"])
print(alloc.to_string(index=False))

print("\nRENT NEEDED (uncovered demand):")
for d in demand.index:
    n = int(short[d].value())
    if n:
        r = demand.loc[d]
        print(f"  Project {r['Project']}: rent {n} x {r['Equipment']}"
              f"  (rentability {r['Rentability']}, criticality {r['Criticality']})")

# verify
assigned = alloc.shape[0]
rented = sum(int(short[d].value()) for d in demand.index)
assert assigned + rented == demand["Qty"].sum()
print(f"\nCheck: {assigned} from fleet + {rented} rented = {demand['Qty'].sum()} demanded. OK")

# ============================================================
# STEP 7 — SAVE THE OUTPUT TO EXCEL
# ============================================================
rent_rows = []
for d in demand.index:
    n = int(short[d].value())
    if n:
        r = demand.loc[d]
        rent_rows.append((r["Project"], r["Equipment"], n, r["Rentability"], r["Criticality"]))

rent_df = pd.DataFrame(rent_rows, columns=["Project", "Equipment", "Qty_To_Rent",
                                           "Rentability", "Criticality"])

with pd.ExcelWriter("Tutorial_Result.xlsx", engine="openpyxl") as xw:
    alloc.to_excel(xw, sheet_name="Allocation", index=False)
    rent_df.to_excel(xw, sheet_name="Rent_Needed", index=False)

print("\nSTEP 7: results written to Tutorial_Result.xlsx "
      "(sheets: Allocation, Rent_Needed)")
