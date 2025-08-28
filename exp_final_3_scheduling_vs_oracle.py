
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


from Framework import StreamScenarioConfig, run_simulation, Tag
from CSCT import CSCT
from ISCT_Oracle import ISCT_Oracle


def generate_dynamic_tag_stream(
    profile: List[Tuple[float, int]],
    mean_residence_time_ms: float,
    std_dev_residence_time_ms: float,
    tag_id_length: int
) -> List[Tag]:
    """
    根据一个分段的负载配置文件，生成一个动态变化的标签流。
    """
    print("--- 正在生成动态标签流... ---")
    tags = []
    current_time_us = 0.0
    for duration_s, rate_tps in profile:
        segment_end_time_us = current_time_us + duration_s * 1e6
        if rate_tps > 0:
            while current_time_us < segment_end_time_us:
                time_to_next_arrival_s = np.random.exponential(1.0 / rate_tps)
                current_time_us += time_to_next_arrival_s * 1e6
                if current_time_us >= segment_end_time_us:
                    break

                entry_time = current_time_us
                residence_time_ms = np.random.normal(
                    mean_residence_time_ms, std_dev_residence_time_ms)
                residence_time_us = max(1, residence_time_ms) * 1e3
                exit_time = entry_time + residence_time_us
                tag_id = ''.join(np.random.choice(
                    ['0', '1']) for _ in range(tag_id_length))
                tags.append(Tag(tag_id, entry_time, exit_time))

    print(f"--- 动态标签流生成完毕，共生成 {len(tags)} 个标签。 ---")
    return tags


def save_results(results: Dict, output_dir: str):
    """
    将仿真结果处理并保存为多个CSV文件。
    """
    print(f"--- 正在将结果保存到目录: {output_dir} ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csct_df = pd.DataFrame(results['CSCT']['time_series_data'])
    CSCT_Oracle_df = pd.DataFrame(results['CSCT_Oracle']['time_series_data'])

    throughput_df = pd.DataFrame({
        'time_s': csct_df['time_ms'] / 1000,
        'CSCT': csct_df['interval_throughput_tps'],
        'CSCT_Oracle': CSCT_Oracle_df['interval_throughput_tps']
    })
    throughput_path = os.path.join(output_dir, 'throughput.csv')
    throughput_df.to_csv(throughput_path, index=False)
    print(f"吞吐量数据已保存至: {throughput_path}")

    backlog_df = pd.DataFrame({
        'time_s': csct_df['time_ms'] / 1000,
        'CSCT': csct_df['unidentified_count_mean'],
        'CSCT_Oracle': CSCT_Oracle_df['unidentified_count_mean']
    })
    backlog_path = os.path.join(output_dir, 'backlog.csv')
    backlog_df.to_csv(backlog_path, index=False)
    print(f"系统积压数据已保存至: {backlog_path}")


def main():
    """
    主函数，负责定义实验参数、运行仿真并调用数据保存函数。
    """

    dynamic_profile = [
        (10.0, 500), (10.0, 1000), (10.0, 2000),
        (10.0, 1000), (10.0, 500)
    ]
    total_simulation_time_s = sum(p[0] for p in dynamic_profile)

    scenario_config = StreamScenarioConfig(
        simulation_duration_s=total_simulation_time_s,
        tag_arrival_rate_per_s=0,
        mean_residence_time_ms=1000.0,
        std_dev_residence_time_ms=50.0,
        tag_id_length=96
    )

    pregenerated_tags = generate_dynamic_tag_stream(
        profile=dynamic_profile,
        mean_residence_time_ms=scenario_config.mean_residence_time_ms,
        std_dev_residence_time_ms=scenario_config.std_dev_residence_time_ms,
        tag_id_length=scenario_config.tag_id_length
    )

    algorithms_to_test = {
        "CSCT": {
            "class": CSCT,
            "config": {
                'initial_q': 11.0, 'priority_weight_size': 1.6, 'priority_weight_depth': 0.12,
            }
        },
        "CSCT_Oracle": {
            "class": ISCT_Oracle,
            "config": {
                'initial_q': 6.0,
                'priority_weight_size': 1,
                'priority_weight_depth': 0.01
            }
        }
    }

    all_results = {}
    for name, algo_info in algorithms_to_test.items():
        print(f"\n--- 正在运行算法: {name} ---")
        result = run_simulation(
            scenario_config=scenario_config,
            algorithm_class=algo_info['class'],
            algorithm_specific_config=algo_info['config'],
            time_series_interval_ms=1000.0,
            pregenerated_tags=pregenerated_tags
        )
        all_results[name] = result
        print(f"--- 算法 {name} 运行完毕 ---")

    output_directory = "exp_final_3_results_vs_oracle"
    save_results(all_results, output_directory)


if __name__ == "__main__":
    main()
    print("\n仿真与数据保存任务完成。")
