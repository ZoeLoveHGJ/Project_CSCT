
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.titlesize': 20
})


RESULTS_BASE_DIR = "exp_final_3_results_vs_oracle"
ALGO_STYLES = {
    'CSCT': {
        'color': 'red', 'linestyle': '-', 'marker': 's', 'label': 'CSCT'
    },
    'CSCT_Oracle': {
        'color': 'blue', 'linestyle': '--', 'marker': '^', 'label': 'CSCT_Oracle'
    }
}
DYNAMIC_PROFILE = [
    (10.0, 500), (10.0, 1000), (10.0, 2000),
    (10.0, 1000), (10.0, 500)
]


def load_data(kpi_filename: str) -> pd.DataFrame:
    """加载指定KPI的CSV结果数据。"""
    csv_path = os.path.join(RESULTS_BASE_DIR, kpi_filename)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        return df.set_index('time_ms')
    else:
        print(f"Error: Result file not found at '{csv_path}'.")
        return pd.DataFrame()


def plot_dynamic_throughput(throughput_data: pd.DataFrame, profile: List[Tuple[float, int]]):
    """使用加载的数据绘制动态吞吐量对比图。"""

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    for algo_name in throughput_data.columns:
        style = ALGO_STYLES.get(algo_name, {})

        ax1.plot(throughput_data.index, throughput_data[algo_name],
                 color=style.get('color'),
                 linestyle=style.get('linestyle'),
                 marker=style.get('marker'),
                 markersize=6,
                 linewidth=2.0,
                 label=f"{style.get('label', algo_name)} Throughput")

    ax1.set_xlabel('Simulation Time (s)')
    ax1.set_ylabel('Real-time Throughput (tags/s)')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_ylim(bottom=0)

    ax1_bg = ax1.twinx()
    time_points = [0] + list(np.cumsum([p[0] for p in profile]))
    rate_points = [p[1] for p in profile] + [profile[-1][1]]
    line_bg, = ax1_bg.step(time_points, rate_points, where='post',
                           color='gray', linestyle=':', alpha=0.8, label='Tag Arrival Rate')

    ax1_bg.set_ylabel('Tag Arrival Rate (tags/s)', color='gray')
    ax1_bg.tick_params(axis='y', labelcolor='gray')
    ax1_bg.set_ylim(bottom=0, top=max(rate_points)*1.15)

    handles, labels = ax1.get_legend_handles_labels()
    handles.append(line_bg)
    labels.append('Tag Arrival Rate')
    ax1.legend(handles=handles, labels=labels, loc='upper left')

    fig.tight_layout(pad=0.5)

    save_path = os.path.join(
        RESULTS_BASE_DIR, "chart_dynamic_throughput_comparison_oracle.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"\nEnlarged content chart successfully saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    if not os.path.isdir(RESULTS_BASE_DIR):
        print(
            f"Error: Results directory '{RESULTS_BASE_DIR}' not found. Please run the data generation script first.")
    else:
        throughput_df = load_data('throughput.csv')

        if not throughput_df.empty:

            throughput_df.index = throughput_df.index / 1000

            plot_dynamic_throughput(throughput_df, DYNAMIC_PROFILE)
        else:
            print("Could not load throughput data. Chart generation aborted.")

    print("\nVisualization task finished.")
