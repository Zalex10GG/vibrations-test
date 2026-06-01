import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

sns.set_theme(style="whitegrid")
palette = sns.color_palette("viridis", n_colors=3)

df = pd.read_csv("convergence.csv")

df["inv_mesh"] = 1.0 / df["mesh_size"]
df = df.sort_values("inv_mesh")

df["error_rel_measures"] = 100.0 * df["frequency_Hz"].diff().abs() / df["frequency_Hz"]
df["error_rel_min"] = 100.0 * (df["frequency_Hz"] - df["frequency_Hz"].iloc[-1]).abs() / df["frequency_Hz"].iloc[-1]

os.makedirs("results/convergency/png", exist_ok=True)
os.makedirs("results/convergency/svg", exist_ok=True)

df.to_csv("results/convergency/convergence_results.csv", index=False)

fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="inv_mesh", y="frequency_Hz", ax=ax1, color=palette[0], s=80, zorder=5)
sns.lineplot(data=df, x="inv_mesh", y="frequency_Hz", ax=ax1, color=palette[0], alpha=0.5)
ax1.set_xlabel("1 / Mesh size [1/m]")
ax1.set_ylabel("Frequency [Hz]")
ax1.set_title("Frequency vs 1/mesh size")
fig1.savefig("results/convergency/png/frequency.png", dpi=150, bbox_inches="tight")
fig1.savefig("results/convergency/svg/frequency.svg", bbox_inches="tight")
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="inv_mesh", y="error_rel_measures", ax=ax2, color=palette[1], s=80, zorder=5)
sns.lineplot(data=df, x="inv_mesh", y="error_rel_measures", ax=ax2, color=palette[1], alpha=0.5)
ax2.set_xlabel("1 / Mesh size [1/m]")
ax2.set_ylabel("Relative error between measures [%]")
ax2.set_title("Error between consecutive measures vs 1/mesh size")
fig2.savefig("results/convergency/png/error_measures.png", dpi=150, bbox_inches="tight")
fig2.savefig("results/convergency/svg/error_measures.svg", bbox_inches="tight")
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="inv_mesh", y="error_rel_min", ax=ax3, color=palette[2], s=80, zorder=5)
sns.lineplot(data=df, x="inv_mesh", y="error_rel_min", ax=ax3, color=palette[2], alpha=0.5)
ax3.set_xlabel("1 / Mesh size [1/m]")
ax3.set_ylabel("Total relative error [%]")
ax3.set_title("Total relative error vs 1/mesh size")
fig3.savefig("results/convergency/png/error_total.png", dpi=150, bbox_inches="tight")
fig3.savefig("results/convergency/svg/error_total.svg", bbox_inches="tight")
plt.close(fig3)

latex_figures_dir = "latex/figures"
os.makedirs(latex_figures_dir, exist_ok=True)
for svg_file in ["frequency.svg", "error_measures.svg", "error_total.svg"]:
    src = os.path.join("results/convergency/svg", svg_file)
    dst = os.path.join(latex_figures_dir, svg_file)
    shutil.copy2(src, dst)

print("Done. CSV and SVGs generated, SVGs copied to latex/figures/.")