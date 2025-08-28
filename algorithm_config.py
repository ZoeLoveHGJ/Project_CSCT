# algorithm_config.py
# -*- coding: utf-8 -*-
"""
Framework-MRS 仿真框架的算法库与样式配置文件。
"""

from ACDQT import ACDQT_Algorithm
from DCT import DCT_Algorithm
from MRSMBA import MRSMBA_Algorithm
from TAD import TAD_Algorithm
from ABP import ABP_Algorithm
from EDRCT import EDRCT_Algorithm
from CSCT import CSCT
from CSCT_Oracle import CSCT_Oracle
from CSCT_Estimator import CSCT_Est
# 2. 定义全局的绘图样式库
PLOT_STYLE_PALETTE = [
    {"color": "#d62728", "linestyle": "-", "marker": "*",
        "linewidth": 3.0, "markersize": 12, "zorder": 10},
    {"color": "#1f77b4", "linestyle": "-", "marker": "o",
        "linewidth": 2.0, "markersize": 8},
    {"color": "#2ca02c", "linestyle": "-.",
        "marker": "s", "linewidth": 2.0, "markersize": 7},
    {"color": "#ff7f0e", "linestyle": ":", "marker": "^"},
    {"color": "#9467bd", "linestyle": "--", "marker": "D"},
    {"color": "#8c564b", "linestyle": "-", "marker": "p"},
    {"color": "#e377c2", "linestyle": "-.", "marker": "x"},
    {"color": "#7f7f7f", "linestyle": ":", "marker": "+"},
    {"color": "#bcbd22", "linestyle": "--", "marker": "h"},
    {"color": "#17becf", "linestyle": "-", "marker": "."},
]

# 3. 定义算法库
ALGORITHM_LIBRARY = {
    'DCT': {
        "class": DCT_Algorithm,
        "config": {},
        "label": "DCT (2019)",
        "style": PLOT_STYLE_PALETTE[1],
    },
    'MRSMBA': {
        "class": MRSMBA_Algorithm,  # 优化版
        "config": {},  # 在这里设置阈值
        "label": "MRSMBA (2019)",
        "style": PLOT_STYLE_PALETTE[2],
    },
    'ACDQT': {
        "class": ACDQT_Algorithm,
        "config": {},
        "label": "ACDQT (2023)",
        "style": PLOT_STYLE_PALETTE[3],
    },
    'TAD': {
        "class": TAD_Algorithm,
        "config": {},
        "label": "TAD (2020)",
        "style": PLOT_STYLE_PALETTE[4],
    },
    'ABP': {
        "class": ABP_Algorithm,
        "config": {},
        "label": "ABP (2023)",
        "style": PLOT_STYLE_PALETTE[5],
    },
    'CSCT': {
        "class": CSCT,  # 优化版
        "config": {
            'initial_q': 11.0,
            'priority_weight_size': 1.8,
            'priority_weight_depth': 0.5,
        },  # 在这里设置阈值
        "label": "CSCT",
        "style": PLOT_STYLE_PALETTE[0],
    },
    'CSCT_Oracle': {
        "class": CSCT_Oracle,  # 优化版
        "config": {
            'initial_q': 11.0,
            'priority_weight_size': 1.60,
            'priority_weight_depth': 0.12,
        },  # 在这里设置阈值
        "label": "ISCT_Oracle (Wrong)",
        "style": PLOT_STYLE_PALETTE[8],
    },
    'CSCT_Est': {
        "class": CSCT_Est,  # 优化版
        "config": {

        },  # 在这里设置阈值
        "label": "CSCT_Est",
        "style": PLOT_STYLE_PALETTE[9],
    },

}
Ohter = {
    # 'ISCT': {
    #     "class": ISCT_Algorithm,
    #     "config": {"T_slice_ms": 50.0},
    #     "label": "ISCT (Ours)",
    #     "style": PLOT_STYLE_PALETTE[0],
    # },
    # 'ACDQT': {
    #     "class": ACDQT,
    #     "config": {"target_loss_rate_th": 0.05, "mean_residence_time_ms": 1000.0},
    #     "label": "ACDQT (2023)",
    #     "style": PLOT_STYLE_PALETTE[1],
    # },
    # 'ISCT_Update1': {
    #     "class": ISCT_Update1, # 优化版
    #     "config": {}, # 在这里设置阈值
    #     "label": "ISCT_Update1 (SOTA)",
    #     "style": PLOT_STYLE_PALETTE[1],
    # },
    # 'ISCT_Update3': {
    #     "class": ISCT_Update3, # 优化版
    #     "config": { "age_threshold_ms": 800,
    #                 "ccq_high_water_mark":5,
    #             }, # 在这里设置阈值
    #     "label": "ISCT_Update3 (Todo)",
    #     "style": PLOT_STYLE_PALETTE[4],
    # },
}
# 4. 定义当前要运行的算法列表
ALGORITHMS_TO_TEST = [
    # 'ISCT_Update2',
    'CSCT',
    # 'ISCT_Update4',
    'DCT',
    'MRSMBA',
    'ACDQT',
    'TAD',
    'ABP',
    #    'CSCT_Est',
]
