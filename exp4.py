import time
import os
import random


import Framework
import Tool
from algorithm_config import ALGORITHM_LIBRARY, ALGORITHMS_TO_TEST


def generate_dynamic_tag_stream(dynamic_config: list, base_config: dict) -> list:
    tags = []
    current_time_us = 0.0
    for duration_s, arrival_rate_per_s in dynamic_config:
        stage_end_time_us = current_time_us + duration_s * 1e6
        if arrival_rate_per_s > 0:
            while current_time_us < stage_end_time_us:
                time_to_next_arrival_s = random.expovariate(arrival_rate_per_s)
                current_time_us += time_to_next_arrival_s * 1e6
                if current_time_us >= stage_end_time_us:
                    break
                entry_time = current_time_us
                residence_time_ms = random.normalvariate(
                    base_config['mean_residence_time_ms'],
                    base_config['std_dev_residence_time_ms']
                )
                residence_time_us = max(1, residence_time_ms) * 1e3
                exit_time = entry_time + residence_time_us
                tag_id = ''.join(random.choice('01')
                                 for _ in range(base_config['tag_id_length']))
                tags.append(Framework.Tag(tag_id, entry_time, exit_time))
    return tags


DYNAMIC_LOAD_PROFILE = [
    (10, 500),
    (10, 1000),
    (10, 2000),
    (10, 1000),
    (10, 500),
]


BASE_SCENARIO_PARAMS = {
    "mean_residence_time_ms": 1000.0,
    "std_dev_residence_time_ms": 50.0,
    "tag_id_length": 96,
}


TOTAL_SIMULATION_DURATION_S = sum(p[0] for p in DYNAMIC_LOAD_PROFILE)


TIME_SERIES_INTERVAL_MS = 1000.0


if __name__ == "__main__":
    start_time = time.time()
    analytics = Tool.SimulationAnalytics()

    print(f"实验四: 综合稳定性分析 - 动态负载场景对比")
    print("-" * 60)
    print(f"总仿真时长: {TOTAL_SIMULATION_DURATION_S} 秒")
    print(f"负载配置: {DYNAMIC_LOAD_PROFILE}")
    print(f"对比算法: {ALGORITHMS_TO_TEST}")
    print("-" * 60)

    for algo_name in ALGORITHMS_TO_TEST:
        print(f"正在为算法 '{algo_name}' 运行仿真...")

        config_for_constructor = {
            "simulation_duration_s": TOTAL_SIMULATION_DURATION_S,
            **BASE_SCENARIO_PARAMS
        }
        scenario_config = Framework.StreamScenarioConfig(
            **config_for_constructor)

        Framework.generate_tag_stream = lambda config: generate_dynamic_tag_stream(
            DYNAMIC_LOAD_PROFILE, BASE_SCENARIO_PARAMS)

        algo_info = ALGORITHM_LIBRARY[algo_name]

        raw_results = Framework.run_simulation(
            scenario_config=scenario_config,
            algorithm_class=algo_info["class"],
            algorithm_specific_config=algo_info["config"],
            time_series_interval_ms=TIME_SERIES_INTERVAL_MS
        )

        scenario_metadata = {
            **config_for_constructor,
            'dynamic_profile': str(DYNAMIC_LOAD_PROFILE)
        }
        analytics.add_run_result(
            raw_results=raw_results,
            scenario_config=scenario_metadata,
            algorithm_name=algo_name,
            run_id=0
        )

    print("\n" + "-" * 60)
    print("所有仿真运行完毕，正在进行数据后处理...")

    output_directory = "exp4_results_full_dynamic_comparison"
    os.makedirs(output_directory, exist_ok=True)

    analytics.plot_results(
        x_axis_key='dynamic_profile',
        algorithm_styles=ALGORITHM_LIBRARY,
        save_path=os.path.join(
            output_directory, "exp4_full_dynamic_charts.png")
    )

    end_time = time.time()
    print("-" * 60)
    print(f"实验四全部完成，总耗时: {end_time - start_time:.2f} 秒。")
    print(f"结果已保存至目录: '{output_directory}'")
    print("-" * 60)
