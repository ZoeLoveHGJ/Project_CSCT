# Exp_Sup_1.py
# -*- coding: utf-8 -*-

import time
import os
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from tqdm import tqdm

import Framework
import Tool
from algorithm_config import ALGORITHM_LIBRARY


NUM_RUNS = 5

TAG_ARRIVAL_RATES = [500, 1000, 1500, 2000, 2500, 3000]

FIXED_SCENARIO_PARAMS = {
    "simulation_duration_s": 10.0,
    "mean_residence_time_ms": 2000.0, 
    "std_dev_residence_time_ms": 500.0,
    "tag_id_length": 96,
    "tag_failure_probability": 0.0, 
    "capture_threshold_db": 0.0,
    "traffic_pattern": 'NORMAL' 
}


VARIANTS = {
    "CSCT (Baseline)": {
        "capture_threshold_db": 0.0,
        "traffic_pattern": 'NORMAL',
        "style": {"color": "#1f77b4", "marker": "o", "linestyle": "-"} # 蓝色
    },
    
    "CSCT (w/ Capture 3dB)": {
        "capture_threshold_db": 3.0, 
        "traffic_pattern": 'NORMAL',
        "style": {"color": "#2ca02c", "marker": "^", "linestyle": "--"} # 绿色
    },
    
    "CSCT (w/ LIFO-Burst)": {
        "capture_threshold_db": 0.0,
        "traffic_pattern": 'BURST_LIFO',
        "style": {"color": "#d62728", "marker": "x", "linestyle": "-."} # 红色
    }
}

TARGET_ALGO_KEY = 'CSCT' 


def run_variant_simulation_worker(task_params: dict) -> dict:

    scenario_config_dict = task_params['scenario_config']
    variant_name = task_params['variant_name'] # 例如 "CSCT (w/ Capture)"
    run_id = task_params['run_id']
    
    base_algo_info = ALGORITHM_LIBRARY[TARGET_ALGO_KEY]
    algo_class = base_algo_info["class"]
    algo_specific_config = deepcopy(base_algo_info["config"])

    scenario_config = Framework.StreamScenarioConfig(**scenario_config_dict)
    
    raw_result_package = Framework.run_simulation(
        scenario_config=scenario_config,
        algorithm_class=algo_class,
        algorithm_specific_config=algo_specific_config
    )
    
    return {
        "raw_results": raw_result_package,
        "scenario_config": scenario_config_dict,
        "algo_name": variant_name, 
        "run_id": run_id
    }

if __name__ == "__main__":
    start_time = time.time()
    analytics = Tool.SimulationAnalytics()

    tasks = []
    
    for rate in TAG_ARRIVAL_RATES:
        for variant_name, variant_params in VARIANTS.items():
            
            current_scenario = FIXED_SCENARIO_PARAMS.copy()
            current_scenario['tag_arrival_rate_per_s'] = rate
            
            for key in ['capture_threshold_db', 'traffic_pattern']:
                if key in variant_params:
                    current_scenario[key] = variant_params[key]
            
            for i in range(NUM_RUNS):
                tasks.append({
                    "scenario_config": current_scenario,
                    "variant_name": variant_name,
                    "run_id": i
                })

    print(f"补充实验一: CSCT 鲁棒性分析")
    print(f"对比组: {list(VARIANTS.keys())}")
    print("-" * 60)
    print(f"总任务数: {len(tasks)} (Rates: {len(TAG_ARRIVAL_RATES)} x Variants: {len(VARIANTS)} x Runs: {NUM_RUNS})")
    
    with Pool(processes=min(len(tasks), cpu_count())) as pool:
        pbar = tqdm(pool.imap_unordered(run_variant_simulation_worker, tasks), total=len(tasks), unit="run")
        
        for res in pbar:
            analytics.add_run_result(
                raw_results=res["raw_results"],
                scenario_config=res["scenario_config"],
                algorithm_name=res["algo_name"], 
                run_id=res["run_id"]
            )

    print("\n" + "-" * 60)
    print("数据处理中...")
    
    output_dir = "exp_sup_1_robustness"
    
    plot_styles = {}
    for name, config in VARIANTS.items():
        plot_styles[name] = {
            "label": name,
            "style": config["style"]
        }

    analytics.save_to_csv(
        x_axis_key='tag_arrival_rate_per_s',
        output_dir=output_dir
    )

    analytics.plot_results(
        x_axis_key='tag_arrival_rate_per_s',
        algorithm_styles=plot_styles,
        save_path=os.path.join(output_dir, "robustness_analysis.png")
    )

    print("-" * 60)
    print(f"实验完成。结果保存在: {output_dir}")