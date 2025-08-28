
# Project-CSCT: 移动RFID抗碰撞算法仿真框架

本项目是 **CSCT** 算法的仿真框架，这是一种我们提出的、基于 EPC C1G2 标准并专为移动 RFID 标签设计的创新性抗碰撞算法。该框架旨在模拟和评估 CSCT 在多种移动环境下，与数个基线算法的性能对比。

## 主要特性

- **遵循 EPC C1G2 标准:** 仿真核心基于 EPCglobal Class 1 Generation 2 标准的规范构建。
- **移动标签仿真:** 专为移动 RFID 标签场景建模，充分考虑了标签移动、驻留时间等因素。
- **多种算法实现:** 框架内不仅实现了我们提出的 CSCT 算法，还包含了多种经典的基线算法，以进行全面的性能比较。
- **丰富的实验脚本:** 提供了一系列实验脚本，可轻松复现针对不同到达率、标签驻留时间、突发性等场景的算法评估。
- **数据分析与可视化:** 包含了将原始仿真数据处理成最终图表和关键性能指标（KPI）的工具和脚本。

## 已实现的算法

### 本文提出的算法
- **CSCT**: 本研究项目中提出的核心算法。
- **CSCT_Estimator**: 使用积压估算技术的 CSCT 算法版本。
- **CSCT_Oracle**: 能够获取完美积压信息的 CSCT 算法版本，作为理论性能的上限参考。

### 基线算法
- **ABP** (Adaptive Binary Splitting)
- **ACDQT** (Adaptive-Correlative Dynamic Query Tree)
- **DCT** (Dynamic-Class-based Traversal)
- **EDRCT** (Efficient Dynamic RFID Tag Collection)
- **MRSMBA** (Mobility-Resistant Slotted-Aloha-based)
- **TAD** (Tag-Amount-Difference-based)

## 项目结构

```
Project-CSCT/
│
├── Framework.py                # 核心仿真框架引擎
├── CSCT.py                     # CSCT 算法的实现
├── MRSMBA.py, DCT.py, etc.     # 各基线算法的实现
│
├── exp0.py, exp1.py, etc.      # 主要的实验执行脚本
├── Analysis.py                 # 用于分析实验结果的辅助脚本
├── Tool.py                     # 实验所用的工具函数
│
├── exp_final_*/                # 存放最终实验结果的目录
├── exp0_results_*/             # 特定实验的结果目录 (包含数据和图表)
│   ├── kpi_*.csv               # 记录关键性能指标 (KPI) 的CSV文件
│   └── *.png, *.pdf            # 根据结果生成的可视化图表
│
└── README.md                   # 本说明文件
```

## 环境配置与安装

1.  **克隆仓库:**
    ```bash
    git clone <your-repository-url>
    cd Project-CSCT
    ```

2.  **安装依赖:**
    建议使用虚拟环境。本项目需要一些常见的科学计算库。
    > **注意:** 项目中未包含 `requirements.txt` 文件。您可能需要手动安装 `pandas`, `matplotlib`, `numpy` 等依赖库。建议创建一个 `requirements.txt` 文件以便于管理。
    ```bash
    pip install pandas matplotlib numpy scipy
    ```

## 如何运行实验

1.  **配置实验参数:**
    打开一个实验文件 (例如 `exp1.py`, `exp2.py`)。
    根据需要修改文件内的参数，如 `TAG_ARRIVAL_RATE` (标签到达率), `SIMULATION_TIME` (仿真时间) 或算法的特定参数。

2.  **执行实验脚本:**
    在终端中运行您选择的实验脚本。
    ```bash
    python exp1.py
    ```

3.  **查看结果:**
    仿真结束后，结果（包括 CSV 数据文件和图表）将会被保存在对应的结果目录中 (例如 `exp1_results_vs_arrival_rate/`)。
