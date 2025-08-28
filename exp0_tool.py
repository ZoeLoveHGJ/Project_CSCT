
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['axes.titlepad'] = 20


def create_final_tuning_report(df: pd.DataFrame, output_dir: str):
    """
    创建最终的参数调优报告：
    采用1x3分面热力图，对比 initial_q = 10, 11, 12 时的性能。
    """
    print("正在生成最终的参数调优报告 (1x3 热力图)...")

    stage2_df = df[df['stage'] == 2].copy()
    if stage2_df.empty:
        print("  -> 警告: 阶段二数据为空，无法生成报告。")
        return

    q_values_to_plot = [10.0, 11.0, 12.0]

    plot_data = stage2_df[stage2_df['initial_q'].isin(q_values_to_plot)]

    if plot_data.empty:
        print(f"  -> 警告: 在阶段二数据中未找到任何 q值为 {q_values_to_plot} 的数据。")
        return

    fig, axes = plt.subplots(1, 3, figsize=(27, 8), sharey=True)

    vmin = plot_data['system_throughput_tps'].min()
    vmax = plot_data['system_throughput_tps'].max()

    new_colormap = "YlGnBu"

    for i, q_val in enumerate(q_values_to_plot):
        ax = axes[i]
        subset_df = plot_data[plot_data['initial_q'] == q_val]

        if not subset_df.empty and subset_df['priority_weight_size'].nunique() > 1 and subset_df['priority_weight_depth'].nunique() > 1:
            pivot = subset_df.pivot_table(
                index='priority_weight_depth', columns='priority_weight_size', values='system_throughput_tps')

            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=new_colormap, linewidths=.5, ax=ax,
                        vmin=vmin, vmax=vmax, cbar=False,
                        annot_kws={"size": 18})

            ax.set_title(f'initial_q = {q_val}',
                         fontweight='bold', fontsize=24)
            ax.set_xlabel('Priority Weight Size',
                          fontweight='bold', fontsize=22)

            if i == 0:
                ax.set_ylabel('Priority Weight Depth',
                              fontweight='bold', fontsize=22)
            else:
                ax.set_ylabel('')
        else:
            ax.text(0.5, 0.5, '数据不足', ha='center',
                    va='center', transform=ax.transAxes)
            ax.set_title(f'initial_q = {q_val}',
                         fontweight='bold', fontsize=24)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = plt.cm.ScalarMappable(
        cmap=new_colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('System Throughput [tps]', fontweight='bold', fontsize=22)

    save_path = os.path.join(output_dir, "final_parameter_tuning_report.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"  -> 最终报告已保存至: {save_path}")
    plt.close(fig)


if __name__ == "__main__":

    input_csv_search = "exp0_tuning_results_all.csv"
    output_dir = "exp0_visual_results"

    if not os.path.exists(input_csv_search):
        print(f"错误: 输入文件 '{input_csv_search}' 未找到。")
        exit()

    os.makedirs(output_dir, exist_ok=True)

    print(f"正在读取参数搜索数据: '{input_csv_search}'...")
    df_search = pd.read_csv(input_csv_search)

    create_final_tuning_report(df_search, output_dir)

    print(f"\n可视化流程完成。最终报告已保存在 '{output_dir}' 目录中。")
