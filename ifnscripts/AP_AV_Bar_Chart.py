import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_barchart(axis=None):
    # Import data
    df = pd.read_csv("AP_AV_Bar_Chart.csv")
    # Prepare data
    df = pd.melt(df, id_vars=['Name'])
    # Plot data
    custom_palette = sns.color_palette("Paired")[1:6]
    ax = sns.barplot(x="Name", y='value', hue='variable', data=df, palette=custom_palette, ax=axis)
    ax.set_yscale('log')
    ax.set_xlabel(None)
    ax.set_ylabel("Value relative to WT")
    plt.xticks(rotation=45)
    plt.tight_layout()
    ax = plt.gca()
    return ax


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")
    ax = plot_barchart()
    plt.savefig(os.path.join(os.getcwd(), "results", "Figures", "Figure_5", "AV_AP_BarChart.pdf"))
