import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def fixed_spacing_floats(s, fmt, nan_value=np.nan):
    """
    Return floats contained in given positions of string s

    Parameters
    ----------
    s : str
        String to parse.
    fmt : list of int
        Alternating gaps and lengths, starting with a gap.
    nan_value : ?, optional
        Value to be returned for failed casts to float. The default is np.nan.

    Raises
    ------
    ValueError
        If len(fmt) < 2.

    Returns
    -------
    values : list of float
        Extracted values.

    """
    if len(fmt) < 2:
        raise ValueError("fmt must have length >= 2")

    values = []
    pos = 0
    for i in range(int(len(fmt) / 2)):
        pos += fmt[i * 2]
        length = fmt[i * 2 + 1]
        value = s[pos : pos + length]
        try:
            value = float(value)
        except ValueError:
            value = nan_value
        values.append(value)
        pos += length

    return values


def read_ptf(filepath):
    """
    Read a Performance Table File for a given aircraft

    Parameters
    ----------
    filepath : str
        File to read.

    Returns
    -------
    ptf : dict
        Extracted data.

    """
    ptf = {}
    with open(filepath, "r", encoding="cp1252") as f:
        lines = f.readlines()

        # Header
        line = lines[7]
        ptf["Vcl1"] = float(line[11:14])
        ptf["Vcl2"] = float(line[15:18])
        ptf["Mcl"] = float(line[23:27])
        ptf["mlo"] = float(line[41:])

        line = lines[8]
        ptf["Vcr1"] = float(line[11:14])
        ptf["Vcr2"] = float(line[15:18])
        ptf["Mcr"] = float(line[23:27])
        ptf["mnom"] = float(line[41:54])
        ptf["hMO"] = float(line[71:])

        line = lines[9]
        ptf["Vdes1"] = float(line[11:14])
        ptf["Vdes2"] = float(line[15:18])
        ptf["Mdes"] = float(line[23:27])
        ptf["mhi"] = float(line[41:])

        # Table
        table = []
        irow = 0
        line = lines[16]
        while line[0] != "=":
            values = fixed_spacing_floats(line, [0, 3, 4, 3, 3, 5, 1, 5, 1, 5, 5, 3, 3, 5, 1, 5, 1, 5, 3, 5, 5, 3, 2, 5, 2, 5])
            table.append(values)
            irow += 1
            line = lines[16 + irow * 2]
        ptf["table"] = pd.DataFrame(
            data=table,
            columns=[
                "FL",
                "Vcr",
                "flo_cr",
                "fnom_cr",
                "fhi_cr",
                "Vcl",
                "ROCDlo_cl",
                "ROCDnom_cl",
                "ROCDhi_cl",
                "fnom_cl",
                "Vdes",
                "ROCDnom_des",
                "fnom_des",
            ],
        ).set_index("FL")

    return ptf

example_ac = read_ptf("./BADA/A320__.PTF")["table"]
example_ac.dropna(inplace=True)

flow_max = max(max(example_ac["fnom_cl"]), max(example_ac["fnom_des"]))
flow_min = min(min(example_ac["fnom_cl"]), min(example_ac["fnom_des"]))
norm = Normalize(vmin=flow_min, vmax=flow_max)
points = plt.scatter(example_ac.index * 100, example_ac["ROCDhi_cl"], label="BADA High Load Climb Rate", c=example_ac["fnom_cl"], cmap="viridis", norm=norm)
plt.scatter(example_ac.index * 100, example_ac["ROCDnom_cl"], label="BADA Nominal Load Climb Rate", c=example_ac["fnom_cl"], cmap="viridis", norm=norm)
plt.scatter(example_ac.index * 100, example_ac["ROCDlo_cl"], label="BADA Low Load Climb Rate", c=example_ac["fnom_cl"], cmap="viridis", norm=norm)
plt.scatter(example_ac.index * 100, example_ac["ROCDnom_des"] * -1, label="BADA Nominal Load Descent Rate",  c=example_ac["fnom_des"], cmap="viridis", norm=norm)
plt.scatter(example_ac.index * 100, np.zeros(len(example_ac.index)), label="BADA Cruise",  c=example_ac["fnom_cr"], cmap="viridis", norm=norm)
cbar = plt.colorbar(points, orientation='horizontal', pad=0.05, aspect=50, label="Fuel Nominal")
plt.xlabel("Altitude (ft)")
plt.ylabel("Altitude Rate (fpm)")
plt.title("BADA Performance")
plt.show()


# Create 2 subplots, stacked vertically, one for climb and one for descent
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Climb plot
ax1_twin = ax1.twinx()
ax1.plot(example_ac.index * 100, example_ac["fnom_cl"]-example_ac["fnom_cr"], label="Additional Climb Fuel Flow", color='tab:blue')
ax1_twin.plot(example_ac.index * 100, example_ac["ROCDnom_cl"], label="Nominal Climb Rate", color='tab:green', linestyle='--')

ax1.set_ylabel("Fuel Flow Rate (kg/hr)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1_twin.set_ylabel("Climb Rate (fpm)", color='tab:green')
ax1_twin.tick_params(axis='y', labelcolor='tab:green')
ax1.set_title("Climb Performance")
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Descent plot
ax2_twin = ax2.twinx()
ax2.plot(example_ac.index * 100, example_ac["fnom_des"]-example_ac["fnom_cr"], label="Additional Descent Fuel Flow", color='tab:blue')
ax2_twin.plot(example_ac.index * 100, -example_ac["ROCDnom_des"], label="Nominal Descent Rate", color='tab:red', linestyle='--')

ax2.set_xlabel("Altitude (ft)")
ax2.set_ylabel("Fuel Flow Rate (kg/hr)", color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2_twin.set_ylabel("Descent Rate (fpm)", color='tab:red')
ax2_twin.tick_params(axis='y', labelcolor='tab:red')
ax2.set_title("Descent Performance")
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')

plt.tight_layout()
plt.show()

# Create 2 subplots, stacked vertically, one for climb and one for descent
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Climb plot
ax1_twin = ax1.twinx()
ax1.plot(example_ac["ROCDnom_cl"], example_ac["fnom_cl"]-example_ac["fnom_cr"], label="Additional Climb Fuel Flow", color='tab:blue')
ax1_twin.plot(example_ac["ROCDnom_cl"], example_ac.index * 100, label="Altitude", color='tab:green', linestyle='--')

ax1.set_xlabel("Rate of Climb (fpm)")
ax1.set_ylabel("Fuel Flow Rate (kg/hr)", color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1_twin.set_ylabel("Altitude (ft)", color='tab:green')
ax1_twin.tick_params(axis='y', labelcolor='tab:green')
ax1.set_title("Climb Performance")
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')

# Descent plot
ax2_twin = ax2.twinx()
ax2.plot(-example_ac["ROCDnom_des"], example_ac["fnom_des"]-example_ac["fnom_cr"], label="Additional Descent Fuel Flow", color='tab:blue')
ax2_twin.plot(-example_ac["ROCDnom_des"], example_ac.index * 100, label="Altitude", color='tab:red', linestyle='--')

ax2.set_xlabel("Rate of Descent (fpm)")
ax2.set_ylabel("Fuel Flow Rate (kg/hr)", color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2_twin.set_ylabel("Altitude (ft)", color='tab:red')
ax2_twin.tick_params(axis='y', labelcolor='tab:red')
ax2.set_title("Descent Performance")
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')

plt.tight_layout()
plt.show()


# Plot fuel flow rate vs vertical rate
plt.figure()
plt.scatter(example_ac["ROCDnom_cl"], example_ac["fnom_cl"], label="Nominal Load Climb", c=example_ac.index, cmap="viridis")
plt.scatter(example_ac["ROCDnom_des"] * -1, example_ac["fnom_des"], label="Nominal Load Descent", c=example_ac.index, cmap="viridis")
plt.scatter(np.zeros(len(example_ac.index)), example_ac["fnom_cr"], label="Cruise", c=example_ac.index, cmap="viridis")
plt.xlabel("Vertical Rate (fpm)")
plt.ylabel("Fuel Flow Rate (kg/hr)")
plt.title("Fuel Flow Rate vs Vertical Rate")
plt.legend()
plt.colorbar(label="Altitude (FL)")
plt.show()

# Plot additional fuel flow rate vs vertical rate
plt.figure()
plt.scatter(example_ac["ROCDnom_cl"], example_ac["fnom_cl"]-example_ac["fnom_cr"], c=example_ac.index, cmap="viridis")
plt.scatter(example_ac["ROCDnom_des"] * -1, example_ac["fnom_des"]-example_ac["fnom_cr"], c=example_ac.index, cmap="viridis")
plt.xlabel("Vertical Rate (fpm)")
plt.ylabel("Fuel Flow Rate (in excess of cruise) (kg/hr)")
plt.title("Additional Fuel Flow Rate (vs cruise) vs Vertical Rate")
plt.colorbar(label="Altitude (FL)")
plt.show()