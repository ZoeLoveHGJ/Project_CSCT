
import time
import os
import numpy as np
from multiprocessing import Pool, cpu_count


from tqdm import tqdm


import Framework
import Tool
from algorithm_config import ALGORITHM_LIBRARY, ALGORITHMS_TO_TEST


NUM_RUNS = 1


TAG_FAILURE_PROBABILITIES = np.linspace(0.0, 0.1, 11)


FIXED_SCENARIO_PARAMS = {
    "simulation_duration_s": 10.0,
    "mean_residence_time_ms": 1000.0,
    "std_dev_residence_time_ms": 50.0,
    "tag_id_length": 96,

    "tag_arrival_rate_per_s": 3000,
}


def run_single_simulation_worker(task_params: dict) -> dict:
    """
    一个独立的工作函数，用于在单个进程中运行一次仿真。
    """

    scenario_config_dict = task_params['scenario_config']
    algo_name = task_params['algo_name']
    run_id = task_params['run_id']

    algo_info = ALGORITHM_LIBRARY[algo_name]
    algo_class = algo_info["class"]
    algo_specific_config = algo_info["config"]

    scenario_config = Framework.StreamScenarioConfig(**scenario_config_dict)

    raw_result_package = Framework.run_simulation(
        scenario_config=scenario_config,
        algorithm_class=algo_class,
        algorithm_specific_config=algo_specific_config
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

    for prob in TAG_FAILURE_PROBABILITIES:
        current_scenario_params = FIXED_SCENARIO_PARAMS.copy()

        current_scenario_params['tag_failure_probability'] = prob

        for algo_name in ALGORITHMS_TO_TEST:
            for i in range(NUM_RUNS):
                tasks.append({
                    "scenario_config": current_scenario_params,
                    "algo_name": algo_name,
                    "run_id": i
                })

    print(f"实验（补充）: 鲁棒性压力测试 - 响应失败率的影响")
    print("-" * 60)
    print(f"固定标签到达率: {FIXED_SCENARIO_PARAMS['tag_arrival_rate_per_s']} tags/s")
    print(
        f"失败率测试范围: {TAG_FAILURE_PROBABILITIES[0]*100:.0f}% to {TAG_FAILURE_PROBABILITIES[-1]*100:.0f}%")
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

    output_directory = "exp_robustness_results_vs_failure_rate"

    analytics.save_to_csv(

        x_axis_key='tag_failure_probability',
        output_dir=output_directory
    )

    analytics.plot_results(

        x_axis_key='tag_failure_probability',
        algorithm_styles=ALGORITHM_LIBRARY,
        save_path=os.path.join(
            output_directory, "exp_robustness_summary_charts.png")
    )

    end_time = time.time()
    print("-" * 60)
    print(f"鲁棒性实验全部完成，总耗时: {end_time - start_time:.2f} 秒。")
    print(f"结果已保存至目录: '{output_directory}'")
    print("-" * 60)
