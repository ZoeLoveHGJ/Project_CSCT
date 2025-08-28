import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

def plot_subplot_a_top5_performance(ax: plt.Axes, df_search: pd.DataFrame):
    """
    在指定的Axes对象上绘制子图(a): Top 5 性能条形图。
    """
    # --- 数据准备 ---
    top5_results = df_search.sort_values('system_throughput_tps', ascending=False).head(5)
    labels = [f"q={row['initial_q']}\nws={row['priority_weight_size']:.2f}\nwd={row['priority_weight_depth']:.2f}" for index, row in top5_results.iterrows()]
    colors = sns.color_palette('viridis_r', n_colors=5)
    
    x_positions = range(len(top5_results))
    throughputs = top5_results['system_throughput_tps'].values

    for i in x_positions:
        bar = ax.bar(i, throughputs[i], color=colors[i], label=labels[i])
        yval = throughputs[i]
        ax.text(i, yval, f'{yval:.2f}', ha='center', va='bottom', fontsize=16, weight='bold')

    # --- 样式与标签 ---
    ax.set_ylabel('System Throughput [tps]', fontweight='bold', fontsize=20)
    ax.set_xlabel('Parameter Combination (initial q, ws, wd)', fontweight='bold', fontsize=20)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=16)
    
    ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax.text(0.5, -0.25, '(a)', transform=ax.transAxes, size=24, weight='bold', ha='center')

def plot_subplot_b_internal_validation(ax: plt.Axes, df_compare_wide: pd.DataFrame):
    df_compare_long = pd.melt(
        df_compare_wide,
        id_vars=['tag_arrival_rate_per_s'],
        value_vars=['ISCT_Oracle', 'ISCT_Update3'],
        var_name='algorithm',
        value_name='RLO_mean'
    )
    
    label_map = {'ISCT_Update3': 'CSCT', 'ISCT_Oracle': 'CSCT_Oracle'}
    df_compare_long['algorithm'] = df_compare_long['algorithm'].map(label_map)

    palette = {"CSCT": "#0072B2", "CSCT_Oracle": "#D55E00"} 
    markers = {"CSCT": "o", "CSCT_Oracle": "s"}
    dashes = {"CSCT": "", "CSCT_Oracle": (4, 1.5)}

    sns.lineplot(
        data=df_compare_long,
        x='tag_arrival_rate_per_s',
        y='RLO_mean',
        hue='algorithm',
        style='algorithm', # 使用style来应用不同的dashes
        markers=markers,
        dashes=dashes,
        palette=palette,
        linewidth=3,      # 增加线条宽度
        markersize=12,    # 增加标记大小
        ax=ax
    )
    
    ax.set_xlabel('Tag Arrival Rate [tags/s]', fontweight='bold', fontsize=20,labelpad=20)
    ax.set_ylabel('Tag Loss Ratio', fontweight='bold', fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(title=None) 
    
    ax.text(0.5, -0.25, '(b)', transform=ax.transAxes, size=24, weight='bold', ha='center')

if __name__ == "__main__":
    
    # --- 1. 定义输入文件和输出目录 ---
    input_csv_search = "exp0_tuning_results_all.csv"
    input_csv_compare = "compare_data.csv"
    output_dir = "exp0_visual_results"
    
    # --- 2. 检查输入文件 ---
    if not os.path.exists(input_csv_search) or not os.path.exists(input_csv_compare):
        print("Error: Input file not found.")
        print(f"Please ensure both '{input_csv_search}' and '{input_csv_compare}' exist.")
        exit()
        
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 3. 读取数据 ---
    print(f"Reading data from '{input_csv_search}' and '{input_csv_compare}'...")
    df_search = pd.read_csv(input_csv_search)
    df_compare = pd.read_csv(input_csv_compare)
    
    
    # --- 4a. 生成并保存子图 (a) ---
    print("Generating plot (a)...")
    fig_a, ax_a = plt.subplots(figsize=(12, 9)) # 为第一个图创建独立的画布
    plot_subplot_a_top5_performance(ax_a, df_search)
    plt.subplots_adjust(bottom=0.25) # 调整布局，为X轴标签留出空间
    save_path_a = os.path.join(output_dir, "report_final_a.pdf")
    plt.savefig(save_path_a, format='pdf', bbox_inches='tight')
    plt.close(fig_a) # 关闭画布，释放内存
    print(f"Plot (a) saved to: '{save_path_a}'")

    # --- 4b. 生成并保存子图 (b) ---
    print("\nGenerating plot (b)...")
    fig_b, ax_b = plt.subplots(figsize=(12, 9)) # 为第二个图创建独立的画布
    plot_subplot_b_internal_validation(ax_b, df_compare)
    plt.subplots_adjust(bottom=0.25) # 调整布局，为X轴标签留出空间
    save_path_b = os.path.join(output_dir, "report_final_b.pdf")
    plt.savefig(save_path_b, format='pdf', bbox_inches='tight')
    plt.close(fig_b) # 关闭画布，释放内存
    print(f"Plot (b) saved to: '{save_path_b}'")

    print(f"\nVisualization complete. Two separate PDF files have been generated in '{output_dir}'.")
