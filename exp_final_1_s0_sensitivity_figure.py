import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 16,          # 基础字号
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20
})

# ==============================================================================
# 1. 绘图参数与配置
# ==============================================================================
RESULTS_BASE_DIR = "exp_final_s0_results_sensitivity"
ARRIVAL_RATES_TO_PLOT = [1500, 3000]
KPI_CONFIG = {
    'average_identification_delay_ms': {
        'csv_suffix': 'kpi_average_identification_delay_ms_vs_S0_value.csv',
        'label': 'Delay (ms)',
        'color': '#ff7f0e'
    },
    'tag_loss_rate': {
        'csv_suffix': 'kpi_tag_loss_rate_vs_S0_value.csv',
        'label': 'Loss Rate',
        'color': '#2ca02c'
    }
}
X_AXIS_KEY = 'estimated_size_per_context'

# ==============================================================================
# 2. 数据加载与合并函数
# ==============================================================================


def load_and_merge_data(rates: list) -> pd.DataFrame:
    """加载所有指定负载下的CSV数据，并合并成一个DataFrame。"""
    all_data = []
    for rate in rates:
        rate_dir = os.path.join(RESULTS_BASE_DIR, f"rate_{rate}_tps")
        if not os.path.isdir(rate_dir):
            print(f"警告: 找不到目录 '{rate_dir}'，跳过此负载。")
            continue

        rate_dfs = []
        for kpi_key, config in KPI_CONFIG.items():
            csv_path = os.path.join(rate_dir, config['csv_suffix'])
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path).set_index(X_AXIS_KEY)
                rate_dfs.append(df)

        if rate_dfs:
            merged_rate_df = pd.concat(rate_dfs, axis=1).reset_index()
            merged_rate_df['load'] = f'{rate} tps'
            all_data.append(merged_rate_df)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)

# ==============================================================================
# 3. 核心绘图函数
# ==============================================================================


def plot_grouped_bar_chart(df: pd.DataFrame):
    """使用合并后的数据绘制带混合标注的分组柱状图。"""

    df = df.set_index(X_AXIS_KEY)
    s0_values = sorted(df.index.unique())
    n_s0 = len(s0_values)

    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    ax2 = ax1.twinx()

    bar_width = 0.20
    index = np.arange(n_s0)
    positions = {
        'Delay_1500': index - 1.5 * bar_width,
        'Delay_3000': index - 0.5 * bar_width,
        'LossRate_1500': index + 0.5 * bar_width,
        'LossRate_3000': index + 1.5 * bar_width,
    }

    baseline = df.loc[2.0]
    baseline_delay_1500 = baseline[baseline['load'] ==
                                   '1500 tps']['average_identification_delay_ms'].iloc[0]
    baseline_delay_3000 = baseline[baseline['load'] ==
                                   '3000 tps']['average_identification_delay_ms'].iloc[0]
    baseline_loss_1500 = baseline[baseline['load']
                                  == '1500 tps']['tag_loss_rate'].iloc[0]
    baseline_loss_3000 = baseline[baseline['load']
                                  == '3000 tps']['tag_loss_rate'].iloc[0]

    delay_config = KPI_CONFIG['average_identification_delay_ms']
    loss_rate_config = KPI_CONFIG['tag_loss_rate']

    delay_1500 = df[df['load'] ==
                    '1500 tps']['average_identification_delay_ms']
    delay_3000 = df[df['load'] ==
                    '3000 tps']['average_identification_delay_ms']
    loss_1500 = df[df['load'] == '1500 tps']['tag_loss_rate']
    loss_3000 = df[df['load'] == '3000 tps']['tag_loss_rate']

    bars_delay_1500 = ax1.bar(positions['Delay_1500'], delay_1500, bar_width,
                              label=f"{delay_config['label']} @ 1500 tps", color=delay_config['color'], alpha=0.9)
    bars_delay_3000 = ax1.bar(positions['Delay_3000'], delay_3000, bar_width,
                              label=f"{delay_config['label']} @ 3000 tps", color=delay_config['color'], alpha=0.6)
    bars_loss_1500 = ax2.bar(positions['LossRate_1500'], loss_1500, bar_width,
                             label=f"{loss_rate_config['label']} @ 1500 tps", color=loss_rate_config['color'], alpha=0.9, hatch='//')
    bars_loss_3000 = ax2.bar(positions['LossRate_3000'], loss_3000, bar_width,
                             label=f"{loss_rate_config['label']} @ 3000 tps", color=loss_rate_config['color'], alpha=0.6, hatch='//')

    def create_hybrid_labels(series, baseline, s0_values, is_loss_rate=False):
        labels = []
        for s0, value in zip(s0_values, series):
            if s0 == 2.0:
                labels.append(
                    f'{value:.1f}' if not is_loss_rate else f'{value:.3f}')
            else:
                if baseline > 1e-9:
                    percent_diff = ((value - baseline) / baseline) * 100
                    labels.append(f'{percent_diff:+.1f}%')
                else:
                    labels.append('')
        return labels

    labels_delay_1500 = create_hybrid_labels(
        delay_1500, baseline_delay_1500, s0_values)
    labels_delay_3000 = create_hybrid_labels(
        delay_3000, baseline_delay_3000, s0_values)
    labels_loss_1500 = create_hybrid_labels(
        loss_1500, baseline_loss_1500, s0_values, is_loss_rate=True)
    labels_loss_3000 = create_hybrid_labels(
        loss_3000, baseline_loss_3000, s0_values, is_loss_rate=True)

    ax1.bar_label(bars_delay_1500, labels=labels_delay_1500,
                  padding=3, fontsize=12)
    ax1.bar_label(bars_delay_3000, labels=labels_delay_3000,
                  padding=3, fontsize=12)
    ax2.bar_label(bars_loss_1500, labels=labels_loss_1500,
                  padding=3, fontsize=12)
    ax2.bar_label(bars_loss_3000, labels=labels_loss_3000,
                  padding=3, fontsize=12)

    ax1.set_xlabel('Initial Estimated Size ($S_0$)')
    ax1.set_ylabel(delay_config['label'], color=delay_config['color'])
    ax2.set_ylabel(loss_rate_config['label'], color=loss_rate_config['color'])

    ax1.set_xticks(index)
    ax1.set_xticklabels(s0_values)

    ax1.tick_params(axis='y', labelcolor=delay_config['color'])
    ax2.tick_params(axis='y', labelcolor=loss_rate_config['color'])

    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, loc='upper center',
               bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)

    ax1.grid(True, which='major', axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(
        RESULTS_BASE_DIR, "chart_s0_sensitivity_single_column.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"\n紧凑版对比图表已成功保存至: {save_path}")
    plt.show()


# ==============================================================================
# 4. 主程序
# ==============================================================================
if __name__ == "__main__":
    if not os.path.isdir(RESULTS_BASE_DIR):
        print(f"错误: 结果根目录 '{RESULTS_BASE_DIR}' 不存在。请先运行实验脚本。")
    else:
        merged_df = load_and_merge_data(ARRIVAL_RATES_TO_PLOT)
        if not merged_df.empty:
            plot_grouped_bar_chart(merged_df)
        else:
            print("未能加载任何数据，无法生成图表。")
    print("\n可视化任务完成。")
