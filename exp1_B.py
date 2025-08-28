import time
import os
import random
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


import Framework
import Tool
from algorithm_config import ALGORITHM_LIBRARY, ALGORITHMS_TO_TEST


NUM_RUNS = 1


BURSTINESS_FACTORS = [1, 2, 5, 10, 15, 20]


FIXED_AVG_ARRIVAL_RATE = 1500


FIXED_SCENARIO_PARAMS = {
    "simulation_duration_s": 20.0,
    "mean_residence_time_ms": 1000.0,
    "std_dev_residence_time_ms": 50.0,
    "tag_id_length": 96,
}


def generate_bursty_tag_stream(
    avg_rate: int,
    burst_factor: int,
    duration_s: float,
    mean_residence_ms: float,
    std_dev_residence_ms: float,
    tag_id_len: int
) -> list:

    if burst_factor <= 1:

        config = Framework.StreamScenarioConfig(
            simulation_duration_s=duration_s,
            tag_arrival_rate_per_s=avg_rate,
            mean_residence_time_ms=mean_residence_ms,
            std_dev_residence_time_ms=std_dev_residence_ms,
            tag_id_length=tag_id_len
        )
        return Framework.generate_tag_stream(config)

    tags = []
    current_time_us = 0.0

    cycle_duration_s = 1.0
    burst_duration_s = cycle_duration_s / burst_factor
    peak_rate = avg_rate * burst_factor

    num_cycles = int(duration_s / cycle_duration_s)

    for _ in range(num_cycles):

        num_tags_in_burst = int(peak_rate * burst_duration_s)
        for _ in range(num_tags_in_burst):

            time_to_arrival_s = random.uniform(0, burst_duration_s)
            entry_time = current_time_us + time_to_arrival_s * 1e6

            residence_time_ms = random.normalvariate(
                mean_residence_ms, std_dev_residence_ms)
            residence_time_us = max(1, residence_time_ms) * 1e3
            exit_time = entry_time + residence_time_us
            tag_id = ''.join(random.choice('01') for _ in range(tag_id_len))
            tags.append(Framework.Tag(tag_id, entry_time, exit_time))

        current_time_us += cycle_duration_s * 1e6

    return tags


def run_single_simulation_worker(task_params: dict) -> dict:

    scenario_config_dict = task_params['scenario_config']
    algo_name = task_params['algo_name']
    run_id = task_params['run_id']

    algo_info = ALGORITHM_LIBRARY[algo_name]
    algo_class = algo_info["class"]
    algo_specific_config = algo_info["config"]

    pregenerated_tags = generate_bursty_tag_stream(
        avg_rate=scenario_config_dict['tag_arrival_rate_per_s'],
        burst_factor=scenario_config_dict['burstiness_factor'],
        duration_s=scenario_config_dict['simulation_duration_s'],
        mean_residence_ms=scenario_config_dict['mean_residence_time_ms'],
        std_dev_residence_ms=scenario_config_dict['std_dev_residence_time_ms'],
        tag_id_len=scenario_config_dict['tag_id_length']
    )

    config_for_framework = scenario_config_dict.copy()
    config_for_framework.pop('burstiness_factor', None)

    scenario_config = Framework.StreamScenarioConfig(**config_for_framework)

    raw_result_package = Framework.run_simulation(
        scenario_config=scenario_config,
        algorithm_class=algo_class,
        algorithm_specific_config=algo_specific_config,

        pregenerated_tags=pregenerated_tags,

        time_series_interval_ms=50.0
    )

    return {
        "raw_results": raw_result_package,
        "scenario_config": scenario_config_dict,
        "algo_name": algo_name,
        "run_id": run_id
    }


if __name__ == "__main__":
    start_time = time.time()
    analytics = Tool.SimulationAnalytics()

    tasks = []
    for burst_factor in BURSTINESS_FACTORS:
        current_scenario_params = FIXED_SCENARIO_PARAMS.copy()
        current_scenario_params['tag_arrival_rate_per_s'] = FIXED_AVG_ARRIVAL_RATE
        current_scenario_params['burstiness_factor'] = burst_factor

        for algo_name in ALGORITHMS_TO_TEST:
            for i in range(NUM_RUNS):
                tasks.append({
                    "scenario_config": current_scenario_params,
                    "algo_name": algo_name,
                    "run_id": i
                })

    print(f"实验 1B: 鲁棒性评估 - 标签到达流突发性的影响")
    print("-" * 60)
    print(f"固定平均到达率: {FIXED_AVG_ARRIVAL_RATE} tags/sec")
    print(f"测试突发度范围: {BURSTINESS_FACTORS}")
    print(f"总计仿真任务数: {len(tasks)}")
    print(f"将使用 {min(len(tasks), cpu_count())} 个CPU核心并行处理...")
    print("-" * 60)

    with Pool(processes=min(len(tasks), cpu_count())) as pool:
        pbar = tqdm(pool.imap_unordered(run_single_simulation_worker,
                    tasks), total=len(tasks), desc="正在运行仿真", unit="run")

        for result_package in pbar:
            analytics.add_run_result(
                raw_results=result_package["raw_results"],
                scenario_config=result_package["scenario_config"],
                algorithm_name=result_package["algo_name"],
                run_id=result_package["run_id"]
            )

    print("\n" + "-" * 60)
    print("所有仿真运行完毕，正在进行数据后处理...")

    output_directory = "exp1B_results_vs_burstiness"

    analytics.save_to_csv(
        x_axis_key='burstiness_factor',
        output_dir=output_directory
    )

    analytics.plot_results(
        x_axis_key='burstiness_factor',
        algorithm_styles=ALGORITHM_LIBRARY,
        save_path=os.path.join(output_directory, "exp1B_summary_charts.png")
    )

    end_time = time.time()
    print("-" * 60)
    print(f"实验 1B 全部完成，总耗时: {end_time - start_time:.2f} 秒。")
    print(f"结果已保存至目录: '{output_directory}'")
    print("-" * 60)
