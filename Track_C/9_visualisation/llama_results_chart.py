import matplotlib.pyplot as plt

# Data
categories = ['MMLU', 'Refusal Rate\n(Validation Split)', 'Refusal Rate\n(Train Split)']
llama_values = [43.3, 61.0, 68.0]
gbrace_values = [42.47, 88.0, 95.5]

# Color-blind–safe palette (Okabe–Ito), softened for wide audience
original_color = "#E69F00"  # orange
gbrace_color = "#0072B2"    # blue

plt.rcParams.update({
    "font.size": 13,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "axes.titlepad": 18,
    "font.family": "DejaVu Sans"
})

fig, ax = plt.subplots(figsize=(12, 7))

bar_width = 0.25
x = range(len(categories))

# Bars with softer fill + darker edge for contrast
bars1 = ax.bar(
    [i - bar_width/2 for i in x],
    llama_values,
    bar_width,
    label='Original Model',
    color=original_color,
    alpha=0.8,
    edgecolor="#333333",
    linewidth=0.8
)

bars2 = ax.bar(
    [i + bar_width/2 for i in x],
    gbrace_values,
    bar_width,
    label='Model Tuned with G-BRACE',
    color=gbrace_color,
    alpha=0.85,
    edgecolor="#333333",
    linewidth=0.8
)

# Value labels
for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 2,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#222222",
            fontweight="semibold"
        )

# Title + "subtitle" for general audience
ax.text(
    0.5, 1.12,
    "Llama 3.2 1B Instruct — Original vs G-BRACE Tuned",
    transform=ax.transAxes,
    ha="center",
    fontsize=20,
    fontweight="bold"
)
ax.text(
    0.5, 1.02,
    "Task performance and refusal rates (%)",
    transform=ax.transAxes,
    ha="center",
    fontsize=13,
    alpha=0.75
)

# Axes
ax.set_ylabel("Percentage (%)", fontsize=14)
ax.set_xticks(list(x))
ax.set_xticklabels(categories, fontsize=13)
ax.set_ylim(0, 115)

# Grid + zero line
ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.3)
ax.axhline(0, color="#444444", linewidth=0.8)

# Legend
legend = ax.legend(
    fontsize=12,
    frameon=True,
    loc='upper left',
    framealpha=0.9
)
legend.get_frame().set_edgecolor("#DDDDDD")

# Background
ax.set_facecolor("#F7F7F7")
fig.patch.set_facecolor("white")

plt.tight_layout()

# Save as high quality SVG
plt.savefig('llama_results_chart.svg', format='svg', bbox_inches='tight', dpi=600)
print("Chart saved to llama_results_chart.svg")

plt.show()
