# -*- coding: utf-8 -*-
# language: english
"""
一个专为 FTRSS 仿真框架设计的分析与可视化工具类。

*** V6.0 - 区间统计可视化增强版 ***

该版本核心升级:
1.  【功能兼容】: 为了与 Framework V9.0 的“区间统计”功能同步，本工具
    现在能够正确解析和处理新的时序数据格式 (如 _mean, _max, _min, _std)。
2.  【可视化升级】: _plot_time_series 方法的绘图逻辑被彻底重构。现在它
    会将每个指标的均值(_mean)绘制为实线，并将最大值(_max)和最小值(_min)
    之间的范围绘制为半透明的填充区域。这种方法能更直观地展示算法在
    一个时间段内的平均性能和稳定性。
3.  【数据导出增强】: 时序数据的CSV导出功能现在会为每一个统计指标
    (如 unidentified_count_mean, unidentified_count_max 等) 都
    生成一个独立的、便于直接绘图的CSV文件。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from functools import reduce

# ==============================================================================
# 中文乱码解决方案
# ==============================================================================
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"警告: 设置'SimHei'字体失败: {e}")
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e2:
        print(f"警告: 设置'Microsoft YaHei'字体也失败: {e2}。图表中的中文可能显示为方块。")

class SimulationAnalytics:
    def __init__(self):
        self.results_data = []

    def add_run_result(self, raw_results: Dict[str, Any], scenario_config: Dict, algorithm_name: str, run_id: int):
        # 此方法逻辑无变化
        kpis = self._calculate_kpis_from_raw(raw_results, scenario_config)
        record = {
            **scenario_config, 
            **raw_results.get('raw_counters', {}),
            **kpis,
            'algorithm_name': algorithm_name, 
            'run_id': run_id
        }
        if 'time_series_data' in raw_results:
            record['time_series_data'] = raw_results['time_series_data']
        self.results_data.append(record)

    def _calculate_kpis_from_raw(self, raw_results: Dict[str, Any], scenario_config: Dict) -> Dict:
        # 此方法逻辑无变化
        all_tags = raw_results['all_tags']
        total_busy_time_us = raw_results['total_busy_time_us']
        simulation_end_time_us = raw_results['simulation_end_time_us']
        raw_counters = raw_results.get('raw_counters', {})
        
        identified_tags = [t for t in all_tags if t.is_identified]
        entered_tags = [t for t in all_tags if t.entry_time_us < simulation_end_time_us]
        lost_tags = [t for t in entered_tags if not t.is_identified]
        
        NEN = len(entered_tags)
        NID = len(identified_tags)
        NLO = len(lost_tags)
        TTS = scenario_config.get('simulation_duration_s', 1.0)
        
        kpis = {}
        kpis['tag_loss_rate'] = NLO / NEN if NEN > 0 else 0
        kpis['system_throughput_tps'] = NID / TTS if TTS > 0 else 0
        total_delay = sum(t.identification_time_us - t.entry_time_us for t in identified_tags)
        kpis['average_identification_delay_ms'] = (total_delay / NID) / 1e3 if NID > 0 else 0
        if identified_tags:
            kpis['time_to_clearance_ms'] = max(t.identification_time_us for t in identified_tags) / 1e3
        else:
            kpis['time_to_clearance_ms'] = 0
        kpis['ratio_of_busyness'] = total_busy_time_us / simulation_end_time_us if simulation_end_time_us > 0 else 0
        
        reader_bits = raw_counters.get('total_reader_bits', 0)
        tag_bits = raw_counters.get('total_tag_bits', 0)
        total_steps = raw_counters.get('total_steps', 0)

        kpis['total_reader_bits'] = reader_bits
        kpis['total_tag_bits'] = tag_bits
        kpis['total_bits'] = reader_bits + tag_bits
        kpis['responses_per_tag'] = total_steps / NID if NID > 0 else 0
        
        return kpis

    def get_results_dataframe(self) -> pd.DataFrame:
        # 此方法逻辑无变化
        return pd.DataFrame(self.results_data) if self.results_data else pd.DataFrame()

    def save_to_csv(self, x_axis_key: str = 'tag_arrival_rate_per_s', output_dir: str = "simulation_results_mobile"):
        # 此方法逻辑无变化, 保持完全兼容
        df = self.get_results_dataframe()
        if df.empty: return
        if x_axis_key not in df.columns:
            print(f"提示: x_axis_key '{x_axis_key}' 在结果中不存在，将跳过宏观KPI的CSV保存。")
            return
        os.makedirs(output_dir, exist_ok=True)
        
        kpi_keys = [
            'tag_loss_rate', 'system_throughput_tps', 'average_identification_delay_ms',
            'time_to_clearance_ms', 'ratio_of_busyness', 'total_reader_bits', 
            'total_tag_bits', 'total_bits', 'responses_per_tag'
        ]
        
        print(f"正在将宏观KPI数据保存到 '{output_dir}' 目录...")
        for key in kpi_keys:
            if key in df.columns:
                try:
                    pivot_df = df.pivot_table(index=x_axis_key, columns='algorithm_name', values=key, aggfunc='mean').reset_index()
                    pivot_df.to_csv(os.path.join(output_dir, f"kpi_{key}.csv"), index=False, float_format='%.4f')
                except Exception as e:
                    print(f"无法为 '{key}' 保存CSV: {e}")
        print("宏观KPI数据保存完成。")

    def plot_results(self, x_axis_key: str = 'tag_arrival_rate_per_s', algorithm_styles: Dict = None, save_path: str = None):
        # 此方法逻辑微调，以传递 output_dir
        df = self.get_results_dataframe()
        if df.empty:
            print("警告: 没有结果可供绘图。")
            return
        
        output_dir = None
        if save_path:
            output_dir = os.path.dirname(save_path)
            os.makedirs(output_dir, exist_ok=True)
        
        if x_axis_key in df.columns and df[x_axis_key].nunique() > 1:
            self._plot_main_kpis(df, x_axis_key, algorithm_styles, save_path)
        
        if 'time_series_data' in df.columns and not df['time_series_data'].isnull().all():
            self._plot_time_series(df, x_axis_key, algorithm_styles, save_path, output_dir)

    def _plot_main_kpis(self, df, x_axis_key, algorithm_styles, save_path):
        # 中文注释: 恢复此方法的完整逻辑
        algorithms = sorted(df['algorithm_name'].unique())
        
        key_to_chinese = {
            'tag_arrival_rate_per_s': '标签到达率 (标签/秒)',
            'mean_residence_time_ms': '平均驻留时间 (毫秒)',
            'priority_weight_depth': '任务深度权重',
            'burstiness_factor': '突发度'
        }
        chinese_x_label = key_to_chinese.get(x_axis_key, x_axis_key.replace('_', ' ').title())

        kpi_config = {
            '标签丢失率 (RLO)': {'key': 'tag_loss_rate', 'ylabel': '丢失率'},
            '系统吞吐率 (THR)': {'key': 'system_throughput_tps', 'ylabel': '吞吐率 (标签/秒)'},
            '平均识别延迟 (DLY)': {'key': 'average_identification_delay_ms', 'ylabel': '延迟 (毫秒)'},
            '清场时间 (TCL)': {'key': 'time_to_clearance_ms', 'ylabel': '时间 (毫秒)'},
            '协议繁忙度 (RBY)': {'key': 'ratio_of_busyness', 'ylabel': '繁忙率'},
            '平均响应次数 (RPT)': {'key': 'responses_per_tag', 'ylabel': '次数/标签'},
        }

        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(24, 14))
        fig.suptitle(f'性能 vs {chinese_x_label}', fontsize=24, fontweight='bold')
        axes = axes.flatten()
        
        for i, (title, config) in enumerate(kpi_config.items()):
            ax = axes[i]
            kpi_key = config['key']
            if kpi_key not in df.columns: continue

            for algo_name in algorithms:
                algo_df = df[df['algorithm_name'] == algo_name]
                grouped_mean = algo_df.groupby(x_axis_key)[kpi_key].mean()
                
                plot_kwargs = {'marker': 'o', 'linestyle': '-', 'linewidth': 2.5, 'markersize': 8}
                label = algo_name
                if algorithm_styles and algo_name in algorithm_styles:
                    plot_kwargs.update(algorithm_styles[algo_name].get("style", {}))
                    label = algorithm_styles[algo_name].get("label", algo_name)
                
                ax.plot(grouped_mean.index, grouped_mean, label=label, **plot_kwargs)

            ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
            ax.set_ylabel(config['ylabel'], fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.legend(fontsize=12, frameon=True, loc='best')
        
        for j in range(len(kpi_config), len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout(rect=[0, 0.03, 1, 0.94], h_pad=3.0)
        if save_path:
            main_save_path = save_path.replace('.png', '_main_kpis.png')
            plt.savefig(main_save_path, dpi=300, bbox_inches='tight')
            print(f"\n主要KPI图表已保存到 {main_save_path}")
        
        plt.show()

    def _plot_time_series(self, df, x_axis_key, algorithm_styles, save_path, output_dir):
        # --- 【核心修改区 V6.0】 ---
        
        # 1. 中文注释: 更新动态指标配置，现在只定义“基础指标”名称
        dynamic_kpi_base_config = {
            'interval_throughput': {'title': '区间吞吐率 vs. 时间', 'ylabel': '吞吐率 (标签/秒)'},
            'unidentified_count': {'title': '实时未识别标签数 vs. 时间', 'ylabel': '未识别标签数'},
            'age_of_oldest_unidentified': {'title': '最长未识别时间 vs. 时间', 'ylabel': '最长延迟 (毫秒)'},
            'internal_backlog_size': {'title': '内部待办任务数 vs. 时间', 'ylabel': '任务数 (个)'},
            'interval_loss_rate': {'title': '区间丢失率 vs. 时间', 'ylabel': '丢失率 (标签/秒)'},
            'channel_efficiency': {'title': '区间信道效率 vs. 时间', 'ylabel': '信道效率'},
        }

        # 2. 确定代表性运行 (逻辑微调)
        sub_df = df
        if x_axis_key in df.columns and df[x_axis_key].nunique() > 1:
            representative_x_val = df[x_axis_key].quantile(0.5, interpolation='nearest')
            sub_df = df[df[x_axis_key] == representative_x_val]
        else:
            representative_x_val = df[x_axis_key].iloc[0] if x_axis_key in df.columns else "Dynamic Profile"

        first_run_ts_data = sub_df.iloc[0].get('time_series_data')
        if not isinstance(first_run_ts_data, list) or not first_run_ts_data:
            print("警告: 找不到用于绘制时序图的数据。")
            return
        
        # 3. 【新增逻辑】: 时序数据重组与导出
        if output_dir:
            print(f"\n正在将时序数据保存到 '{output_dir}' 目录...")
            all_ts_keys = first_run_ts_data[0].keys()
            for kpi_key in all_ts_keys:
                if kpi_key == 'time_ms': continue # 跳过时间戳列
                
                all_algo_ts_data = []
                for algo_name in sub_df['algorithm_name'].unique():
                    run_data = sub_df[sub_df['algorithm_name'] == algo_name].iloc[0]
                    time_series = run_data.get('time_series_data')
                    if isinstance(time_series, list) and time_series:
                        ts_df = pd.DataFrame(time_series)
                        if kpi_key in ts_df.columns:
                            algo_df = ts_df[['time_ms', kpi_key]].rename(columns={kpi_key: algo_name})
                            all_algo_ts_data.append(algo_df)
                
                if all_algo_ts_data:
                    merged_df = reduce(lambda left, right: pd.merge(left, right, on='time_ms', how='outer'), all_algo_ts_data)
                    merged_df = merged_df.sort_values(by='time_ms').reset_index(drop=True)
                    output_filename = os.path.join(output_dir, f"timeseries_{kpi_key}.csv")
                    merged_df.to_csv(output_filename, index=False, float_format='%.4f')
            print("时序数据保存完成。")

        # 4. 【重构逻辑】: 动态创建图表布局
        available_base_kpis = [k for k in dynamic_kpi_base_config if f"{k}_mean" in first_run_ts_data[0] or f"{k}_tps" in first_run_ts_data[0]]
        if not available_base_kpis:
            print("警告: 时序数据中没有可绘制的统计指标。")
            return
            
        num_kpis = len(available_base_kpis)
        ncols = 2
        nrows = (num_kpis + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 6 * nrows), sharex=True)
        axes = np.array(axes).flatten()
        fig.suptitle(f'动态性能分析 @ {x_axis_key} = {representative_x_val}', fontsize=24, fontweight='bold')

        # 5. 【重构逻辑】: 循环绘制所有算法和所有可用指标
        for i, base_kpi_key in enumerate(available_base_kpis):
            ax = axes[i]
            config = dynamic_kpi_base_config[base_kpi_key]

            for algo_name in sub_df['algorithm_name'].unique():
                run_data = sub_df[sub_df['algorithm_name'] == algo_name].iloc[0]
                time_series = run_data.get('time_series_data')
                if isinstance(time_series, list) and time_series:
                    ts_df = pd.DataFrame(time_series)
                    ts_df['time_sec'] = ts_df['time_ms'] / 1000
                    
                    # 中文注释: 确定要绘制的列名
                    mean_key = f"{base_kpi_key}_mean"
                    min_key = f"{base_kpi_key}_min"
                    max_key = f"{base_kpi_key}_max"
                    # 中文注释: 兼容吞吐率和丢失率这种没有统计后缀的指标
                    if base_kpi_key.endswith('_tps'):
                        mean_key = base_kpi_key
                    elif base_kpi_key.startswith('interval_'):
                         mean_key = f"{base_kpi_key}_tps"

                    if mean_key in ts_df.columns:
                        plot_kwargs = {'linewidth': 2.0}
                        label = algo_name
                        color = None
                        if algorithm_styles and algo_name in algorithm_styles:
                            style = algorithm_styles[algo_name].get("style", {})
                            plot_kwargs.update(style)
                            plot_kwargs.pop('marker', None) # 时序图通常不需要标记点
                            label = algorithm_styles[algo_name].get("label", algo_name)
                            color = style.get('color')

                        # 中文注释: 绘制均值曲线
                        ax.plot(ts_df['time_sec'], ts_df[mean_key], label=label, **plot_kwargs)

                        # 中文注释: 如果存在最大/最小值，则绘制填充区域
                        if min_key in ts_df.columns and max_key in ts_df.columns:
                            ax.fill_between(ts_df['time_sec'], ts_df[min_key], ts_df[max_key], color=color, alpha=0.2)

            ax.set_title(config['title'], fontsize=16, fontweight='bold', pad=12)
            ax.set_ylabel(config['ylabel'], fontsize=14)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            ax.legend(fontsize=12, frameon=True, loc='best')
            if i >= (nrows - 1) * ncols:
                ax.set_xlabel('仿真时间 (秒)', fontsize=14)

        # 6. 删除多余的空子图 (逻辑无变化)
        for j in range(num_kpis, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=4.0, w_pad=3.0)
        
        # 7. 保存图表 (逻辑无变化)
        if save_path:
            ts_save_path = save_path.replace('.png', '_time_series.png')
            plt.savefig(ts_save_path, dpi=300, bbox_inches='tight')
            print(f"\n动态时序图表已保存到 {ts_save_path}")
        
        plt.show()

# RfidUtils 类无变化
class RfidUtils:
    @staticmethod
    def get_collision_info(tag_ids: List[str]) -> tuple[str, List[int]]:
        if not tag_ids: return "", []
        if len(tag_ids) == 1: return tag_ids[0], []
        prefix = os.path.commonprefix(tag_ids)
        id_len = len(tag_ids[0])
        collision_positions = []
        for i in range(len(prefix), id_len):
            if len({tag[i] for tag in tag_ids}) > 1:
                collision_positions.append(i)
        return prefix, collision_positions
