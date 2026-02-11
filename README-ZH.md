# CSCT：上下文调度碰撞树

> **一种面向移动RFID标签识别的可靠抗碰撞协议**

[![期刊](https://img.shields.io/badge/期刊-Computer%20Networks-blue)](https://www.sciencedirect.com/journal/computer-networks)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.comnet.2026.112091-green)](https://doi.org/10.1016/j.comnet.2026.112091)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-orange)](LICENSE)

本仓库包含以下论文的源代码与仿真框架：

**"Context-Scheduled Collision Tree: A Reliable Anti-collision Protocol for Mobile RFID Tags Identification"**

发表于 *Computer Networks*（在线发表日期：2026年2月10日）

---

## 目录

- [研究背景](#研究背景)
- [提出的方法](#提出的方法)
- [项目结构](#项目结构)
- [已实现的算法](#已实现的算法)
- [关键性能指标](#关键性能指标)
- [快速开始](#快速开始)
- [运行实验](#运行实验)
- [关键结论](#关键结论)
- [引用](#引用)

---

## 研究背景

在传统RFID系统中，标签被假定在被识别之前一直静止于读写器的询问区域内。然而，在现实应用场景中，如**仓储物流**、**传送带盘点**和**车辆门禁控制**等，RFID标签附着于移动物体上，**持续进出**读写器场。这种移动性带来了严峻挑战：

- **标签丢失**：标签可能在成功识别之前就已离开读写器场。
- **动态积压**：未识别标签的数量随时间不可预测地波动。
- **时间紧迫性**：每个标签的驻留时间有限，为识别设定了隐式截止期限。

现有的基于树或基于ALOHA的抗碰撞协议主要针对静态场景设计，无法充分应对上述挑战，导致在移动条件下出现较高的标签丢失率和较低的系统吞吐量。

## 提出的方法

我们提出了 **CSCT（Context-Scheduled Collision Tree，上下文调度碰撞树）**，一种专为移动RFID标签识别设计的新型抗碰撞协议。CSCT采用**两阶段架构**：

### 阶段 A — 发现阶段
使用时隙ALOHA帧快速发现场内的活跃标签集合。新入场的标签根据其ID的哈希值进行响应，碰撞被记录为**碰撞上下文（Collision Context）**——一种结构化的元数据，捕获碰撞标签集合及其关联的前缀信息。

### 阶段 B — 解析阶段
碰撞上下文在**上下文碰撞队列（Context-Collision Queue, CCQ）**中进行管理。CCQ是一个优先级队列，基于可配置的优先级函数调度解析顺序：

```
Priority = α × EstimatedSize − β × PrefixDepth
```

该调度策略优先处理较大的碰撞组（包含更多面临丢失风险的标签），同时偏好较浅的树节点（可更高效地解析）。

### 核心创新
1. **上下文感知调度** — 碰撞解析根据每个碰撞上下文的估计紧迫性和规模进行动态优先级排序。
2. **两阶段解耦架构** — 将新标签的发现过程与已知碰撞的解析过程分离，实现对持续到达流量的不间断处理。
3. **物理层鲁棒性建模** — 集成了随机信道差错和捕获效应的仿真，确保评估结果的真实性。

## 项目结构

```
Project_CSCT/
│
├── Framework.py               # 核心仿真引擎 (FTRSS)
│                               #   - EPC C1G2 物理层参数
│                               #   - 移动标签流生成（泊松分布 & 突发/LIFO模式）
│                               #   - 事件驱动仿真循环
│                               #   - 信道差错 & 捕获效应建模
│
├── CSCT.py                    # ★ 提出的 CSCT 算法
├── CSCT_Estimator.py          # CSCT 变体：带积压估算机制
├── CSCT_Oracle.py             # CSCT 变体：拥有完美积压信息（理论上界）
│
├── ABP.py                     # 基线算法：自适应二进制分裂协议
├── ACDQT.py                   # 基线算法：自适应相关动态查询树
├── DCT.py                     # 基线算法：动态碰撞树
├── MRSMBA.py                  # 基线算法：抗移动时隙ALOHA
├── TAD.py                     # 基线算法：基于标签量差异
│
├── algorithm_config.py        # 算法注册表、配置与绘图样式
├── Tool.py                    # KPI计算与可视化工具类
├── Analysis.py                # 跨实验结果对比分析器
│
├── exp1.py                    # 实验1：性能 vs. 标签到达率
├── exp1_B.py                  # 实验1B：性能 vs. 突发性
├── exp1_C.py                  # 实验1C：到达率补充分析
├── exp2.py                    # 实验2：性能 vs. 标签驻留时间
├── exp4.py                    # 实验4：全动态对比
├── exp0.py                    # 实验0：参数调优与灵敏度分析
├── Exp_Sup_1/2/3.py           # 补充实验脚本
├── exp_final_*.py             # 最终定稿实验脚本
│
├── exp*_results_*/            # 输出目录（CSV数据 + 图表）
├── Test_Run.py                # 快速单次运行与调试脚本
├── requirements.txt           # Python 依赖列表
└── .gitignore
```

## 已实现的算法

### 本文提出

| 算法 | 说明 |
|---|---|
| **CSCT** | 上下文调度碰撞树（本文工作） |
| **CSCT_Estimator** | 带在线积压估算机制的CSCT |
| **CSCT_Oracle** | 拥有完美（oracle）积压信息的CSCT — 理论上界 |

### 基线算法

| 算法 | 全称 | 参考年份 |
|---|---|---|
| **DCT** | Dynamic Collision Tree | 2019 |
| **MRSMBA** | Mobility-Resistant Slotted Multiple Binary-split Algorithm | 2019 |
| **TAD** | Tag-Amount-Difference-based | 2020 |
| **ABP** | Adaptive Binary-splitting Protocol | 2023 |
| **ACDQT** | Adaptive-Correlative Dynamic Query Tree | 2023 |

## 关键性能指标

仿真框架使用以下 KPI 评估所有算法的性能：

| 指标 | 符号 | 说明 |
|---|---|---|
| **标签丢失率** | RLO | 未被识别即离场的标签所占比例 |
| **系统吞吐率** | THR | 每秒成功识别的标签数（tags/s） |
| **平均识别延迟** | DLY | 标签从到达场内到成功识别的平均时间（ms） |
| **清场时间** | TCL | 识别一批同时到达标签所需的时间（ms） |
| **协议繁忙度** | RBY | 读写器信道被占用的时间比例 |
| **通信开销** | — | 读写器和标签发送的总比特数 |

## 快速开始

### 环境要求

- Python 3.8 及以上版本
- pip（Python 包管理器）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/ZoeLoveHGJ/Project_CSCT.git
cd Project_CSCT

# （推荐）创建虚拟环境
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# 安装依赖
pip install -r requirements.txt
```

### 依赖说明

| 依赖包 | 用途 |
|---|---|
| `numpy` | 数值计算 |
| `pandas` | 数据处理与CSV导出 |
| `matplotlib` | 结果可视化 |
| `scipy` | 科学计算工具 |

## 运行实验

### 快速验证

运行以下命令进行框架的快速功能验证：

```bash
python Test_Run.py
```

### 主要实验

每个实验脚本均为独立、自包含的，可直接运行：

```bash
# 实验1：标签到达率的影响
python exp1.py

# 实验2：标签驻留时间的影响
python exp2.py

# 实验4：全动态场景对比
python exp4.py

# 实验1B：突发流量的影响
python exp1_B.py
```

### 参数调优

探索CSCT的参数灵敏度：

```bash
python exp0.py
```

### 输出结果

每个实验完成后，结果将保存至对应的输出目录：
- **CSV 文件**：原始KPI数据（如 `kpi_tag_loss_rate.csv`、`kpi_system_throughput_tps.csv`）
- **PNG/PDF 图表**：自动生成的对比图

### 跨实验对比分析

比较CSCT在所有实验中相对基线算法的性能提升：

```bash
python Analysis.py
```

## 关键结论

在典型的移动RFID场景下，CSCT展现了以下优势：

- **显著降低标签丢失率** — CSCT的上下文感知调度确保高优先级碰撞组在标签离场前得到解析。
- **更高的系统吞吐率** — 两阶段架构通过发现与解析的重叠执行最大化了信道利用率。
- **在突发流量下的鲁棒性** — CSCT凭借动态Q值调整和优先级调度，能够很好地适应标签到达的突发激增。
- **接近Oracle的性能** — CSCT（带估算）与CSCT_Oracle（完美信息）之间的性能差距很小，验证了调度启发式策略的有效性。

## 引用

如果本代码对您的研究有所帮助，请引用我们的论文：

```bibtex
@article{ZHOU2026112091,
title = {Context-Scheduled Collision Tree: A Reliable Anti-collision Protocol for Mobile RFID Tags Identification},
journal = {Computer Networks},
pages = {112091},
year = {2026},
doi = {https://doi.org/10.1016/j.comnet.2026.112091},
author = {Hongquan Zhou and Xiaolin Jia and Zhong Du and Yajun Gu and Hong Yang},
}
```

## 许可证

本项目以学术研究为目的开源。具体条款请参阅 [LICENSE](LICENSE) 文件。

---

**[English README](README-EN.md)**
