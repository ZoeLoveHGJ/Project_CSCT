import time
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
import heapq


import Framework
import Tool
from CSCT import CSCT, CollisionContext


NUM_RUNS = 1

S0_VALUES_TO_TEST = [2.0, 3.0, 4.0, 5.0, 8.0]

ARRIVAL_RATES_TO_TEST = [1500, 3000]


FIXED_SCENARIO_PARAMS = {
    "simulation_duration_s": 10.0,
    "mean_residence_time_ms": 1000.0,
    "std_dev_residence_time_ms": 50.0,
    "tag_id_length": 96,
}


def run_single_simulation_worker(task_params: dict) -> dict:
    """一个独立的工作函数，用于在单个进程中运行一次仿真。"""
    scenario_config_dict = task_params['scenario_config']
    algo_specific_config = task_params['algo_config']
    run_id = task_params['run_id']

    scenario_config = Framework.StreamScenarioConfig(**scenario_config_dict)

    raw_result_package = Framework.run_simulation(
        scenario_config=scenario_config,
        algorithm_class=CSCT,
        algorithm_specific_config=algo_specific_config
    )

    return {
        "raw_results": raw_result_package,
        "scenario_config": scenario_config_dict,

        "algo_name": f"CSCT_S0={algo_specific_config['estimated_size_per_context']}",
        "run_id": run_id
    }


if __name__ == "__main__":
    start_time = time.time()
    analytics = Tool.SimulationAnalytics()

    tasks = []
    for rate in ARRIVAL_RATES_TO_TEST:
        for s0_val in S0_VALUES_TO_TEST:
            current_scenario_params = FIXED_SCENARIO_PARAMS.copy()
            current_scenario_params['tag_arrival_rate_per_s'] = rate

            algo_config = {
                'initial_q': 11.0,
                'priority_weight_size': 1.0,
                'priority_weight_depth': 1.0,
                'estimated_size_per_context': s0_val
            }

            for i in range(NUM_RUNS):
                tasks.append({
                    "scenario_config": current_scenario_params,
                    "algo_config": algo_config,
                    "run_id": i
                })

    print("实验一: CSCT核心参数S0敏感性分析")
    print("-" * 60)
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

    output_directory = "exp_final_s0_results_sensitivity"
    os.makedirs(output_directory, exist_ok=True)

    full_df = analytics.get_results_dataframe()
    for rate in ARRIVAL_RATES_TO_TEST:

        rate_df = full_df[full_df['tag_arrival_rate_per_s'] == rate].copy()
        if rate_df.empty:
            continue

        rate_df['estimated_size_per_context'] = rate_df['algorithm_name'].apply(
            lambda x: float(x.split('=')[1]))

        sub_dir = os.path.join(output_directory, f"rate_{rate}_tps")
        os.makedirs(sub_dir, exist_ok=True)

        kpi_keys_to_save = ['system_throughput_tps',
                            'tag_loss_rate', 'average_identification_delay_ms']

        print(f"\n正在保存负载 {rate} tps 的结果至目录: '{sub_dir}'")
        for key in kpi_keys_to_save:

            pivot_df = rate_df.pivot_table(
                index='estimated_size_per_context', values=key, aggfunc='mean').reset_index()
            output_filename = os.path.join(
                sub_dir, f"kpi_{key}_vs_S0_value.csv")
            pivot_df.to_csv(output_filename, index=False, float_format='%.4f')
            print(f"  - 已保存: {os.path.basename(output_filename)}")

        rate_analytics = Tool.SimulationAnalytics()
        rate_analytics.results_data = rate_df.to_dict('records')

        print(f"  - 绘图数据已生成在 '{sub_dir}' 目录中")

    end_time = time.time()
    print("-" * 60)
    print(f"实验一全部完成，总耗时: {end_time - start_time:.2f} 秒。")
    print(f"结果已保存至目录: '{output_directory}'")
    print("-" * 60)