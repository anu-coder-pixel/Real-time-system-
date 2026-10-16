# Context-Aware-Slack-Driven Dynamic Task Scheduling (CASDS) for Automotive Cyber-Physical Systems

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![IEEE TII](https://img.shields.io/badge/IEEE%20TII-Accepted%202026-brightgreen.svg)](https://doi.org/10.1109/TII.2026.3704380)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](app.py)

Official comprehensive implementation and replication suite of the research paper:

> **"Context-Aware-Slack-Driven Dynamic Task Scheduling for Automotive Cyber-Physical Systems"**  
> *Chuang Zhang, Sifa Zheng, Chaoyi Chen, Haoran Li, Hao Cheng, Qing Xu, and Keqiang Li*  
> **IEEE Transactions on Industrial Informatics (2026)**. DOI: `10.1109/TII.2026.3704380`.

---

## 📌 Executive Summary

Modern Autonomous Vehicles (AVs) transition from distributed Microcontroller Units (MCUs/ECUs) under AUTOSAR Classic Platform (CP) to centralized vehicular computing platforms under AUTOSAR Adaptive Platform (AP). This introduces major challenges:
1. **Heterogeneity**: CPUs, GPUs, and specialized ASICs with diverse execution profiles and bus latencies.
2. **Parallelism & Dynamics**: Concurrent intelligent applications released dynamically and unpredictably according to road traffic conditions.
3. **Quality of Service (QoS)**: Measured by Deadline Satisfaction Ratio (DSR).
4. **Mixed Criticality**: Safety-critical tasks vs. non-safety tasks (e.g. ISO 26262 ASIL levels $S_1, S_2, S_3, S_4$).
5. **Precedence Dependence**: Multi-task Directed Acyclic Graphs (DAGs) with communication transfer latencies.

**CASDS** solves the limitations of prior state-of-the-art schedulers:
- Prior approaches (ADS, ASDYS) relied on crude mode-switching or task dropping, starving low-criticality tasks.
- Radix sorting (HetSched) created critical task congestion and blocked non-critical tasks.
- Static subdeadline schedulers (TD_ADS, ULS) failed to adapt to dynamic runtime resource bottlenecks.

CASDS introduces **Context-Aware Slack (CAS)** and **TOPSIS + Entropy Weight Adaptive Priority**, balancing task criticality and real-time urgency to dramatically boost system DSR, effective throughput, and fault robustness.

---

## 📐 Mathematical Formulation

### 1. Task Sorting within Application (HEFT Upward Rank, Eq. 7)
For each task $v_i^a \in V^a$ in application $A_a$:
$$\bar{w}_i^a = \frac{1}{|\mathcal{P}_{valid}|} \sum_{p_k \in \mathcal{P}_{valid}} w_{i,k}^a$$
$$rank_u(v_i^a) = \bar{w}_i^a + \max_{v_j^a \in succ(v_i^a)} \left( rank_u(v_j^a) + \bar{c}_{i,j}^a \right)$$
Tasks are sorted in descending order of $rank_u(v_i^a) \to Q_o^a$.

---

### 2. Context-Aware Slack (CAS) & Urgency (Eq. 8, 9, 10)
- **Remaining Time**:
  $$T_{rem}^{a,i} = T_{arr}^a + T_{rd}^a - t$$
- **Predictive Least Completion Time (PLCT)**:
  $$ERT(v_i^a, p_k) = \max_{v_j \in pred(v_i)} \left\{ \max(\hat{T}_{ft}^{a,j}, t) + \hat{c}_{i,j}^a \right\}$$
  $$EST(v_i^a, p_k, \mathcal{P}_t^s) = \max\left\{ \Lambda(p_k, \mathcal{P}_t^s), ERT(v_i^a, p_k) \right\}$$
  $$EFT(v_i^a, p_k, \mathcal{P}_t^s) = w_{i,k}^a + EST(v_i^a, p_k, \mathcal{P}_t^s)$$
  $$PLCT(Q_u^a, \mathcal{P}_t^s) = \max_{v_i^a \in Q_u^a} \left\{ \min_{p_k \in \mathcal{P}} EFT(v_i^a, p_k, \mathcal{P}_t^s) \right\} - t$$
- **Context-Aware Slack**:
  $$CAS(v_i^a, \mathcal{P}_t^s) = T_{rem}^{a,i} - PLCT(Q_u^a, \mathcal{P}_t^s)$$
  $$U_i^a = CAS(v_i^a, \mathcal{P}_t^s)$$
  *Physical Meaning*: Maximum available time margin for scheduling before deadline violation. If $U_i^a < 0$, the deadline is unachievable under any schedule; execution is deferred to prevent domino congestion.

---

### 3. Adaptive Priority via TOPSIS & Entropy Weight Method (Eq. 11, 12, 13)
Balances criticality $\mathcal{C}_i^a$ and urgency $-U_i^a$:
- **Min-Max Normalization**:
  $$x_{n,m}^{norm} = \frac{x_{n,m} - x_{\min,m}}{x_{\max,m} - x_{\min,m}}$$
- **Entropy Weight**:
  $$p_{n,m} = \frac{x_{n,m}^{norm}}{\sum_{n'=1}^N x_{n',m}^{norm}}, \quad \varepsilon_m = -\frac{1}{\ln N} \sum_{n=1}^N p_{n,m} \ln p_{n,m}$$
  $$\omega_m = \frac{1 - \varepsilon_m}{\sum_{m'=1}^M (1 - \varepsilon_{m'})}, \quad x_{n,m}^{wt} = \omega_m \cdot x_{n,m}^{norm}$$
- **Relative Closeness Score**:
  $$Z_m^+ = \max_n \{x_{n,m}^{wt}\}, \quad Z_m^- = \min_n \{x_{n,m}^{wt}\}$$
  $$Sim_n^+ = \sqrt{\sum_{m=1}^M (x_{n,m}^{wt} - Z_m^+)^2}, \quad Sim_n^- = \sqrt{\sum_{m=1}^M (x_{n,m}^{wt} - Z_m^-)^2}$$
  $$P_i^a = Sim_n = \frac{Sim_n^-}{Sim_n^- + Sim_n^+}$$

---

### 4. Task-PE Allocation (Eq. 14, 15)
$$p_{tg}^{a,i} = \arg\min_{p_k \in \mathcal{P}} \left\{ EFT(v_i^a, p_k, \mathcal{P}_t^s) \right\}$$
$$p_{alc}^{a,i} = \begin{cases} p_{tg}^{a,i} & \text{if } \Lambda(p_{tg}^{a,i}, \mathcal{P}_t^s) = t \\ \text{Null} & \text{otherwise} \end{cases}$$
Tasks allocated with $p_{alc}^{a,i} \neq \text{Null}$ start at $EST(v_i^a, p_{alc}^{a,i}, \mathcal{P}_t^s)$. Tasks with $\text{Null}$ wait in the ready queue to re-engage at the next scheduling event.

---

## 🗂️ Project Directory Structure

```text
real_time_project/
├── app.py                           # High-Tech Streamlit Web Dashboard & HUD
├── README.md                        # Documentation and research paper guide
├── casds/
│   ├── __init__.py
│   ├── models/
│   │   ├── task.py                  # Task node model, execution profiles, status lifecycle
│   │   ├── platform.py              # Heterogeneous PEs (CPU, GPU, ASIC), state tracking Lambda, comm matrix
│   │   └── dag.py                   # DAG application model, HEFT/AEFT ranking, subdeadlines
│   ├── benchmarks/
│   │   ├── adsuite_benchmark.py     # Table I 13-PE hardware platform & 4 ADSuite DAG types
│   │   ├── selfpilot_benchmark.py   # Table IV real vehicle platform & SelfPilot autonomous driving DAG
│   │   └── trace_generator.py       # Poisson arrivals, mixed criticality, noise & failure injection
│   ├── schedulers/
│   │   ├── base_scheduler.py        # Base dynamic scheduler with unified EFT allocation
│   │   ├── casds_scheduler.py       # Proposed CASDS scheduler (PLCT, CAS, TOPSIS, EFT)
│   │   ├── baselines.py             # ADS (2017), ASDYS (2021), HetSched (2021), TD_ADS (2024), ULS (2025)
│   │   └── ablation_schedulers.py   # Table II ablation variants (T_rem, T-Ddl, w/o Crit, Multikey, AEFT)
│   ├── simulation/
│   │   ├── simulator.py             # Discrete-event simulation engine
│   │   └── metrics.py               # DSR(S), DSR(S_crit), Makespan, Utilization, Effective Throughput
│   ├── experiments/
│   │   ├── exp1_simulation_benchmark.py    # Figure 5 reproduction (DSR vs lambda and R_crit)
│   │   ├── exp2_system_efficiency.py        # Figure 6 reproduction (Makespan, Util, Throughput)
│   │   ├── exp3_ablation_study.py           # Table II reproduction (Ablation study)
│   │   ├── exp4_robustness_study.py         # Figures 7 & 8 reproduction (Delay & PE failures)
│   │   ├── exp5_scalability_study.py        # Table III reproduction (4 ASIL criticality levels)
│   │   ├── exp6_real_hardware_selfpilot.py  # Figure 10 reproduction (SelfPilot platform)
│   │   └── run_all_experiments.py           # Master runner executing all experiments & plots
│   └── visualization/
│       └── plot_generator.py        # IEEE TII publication-quality plot generator
├── tests/
│   └── test_basic.py                # Unit test suite verifying models, algorithms, and schedulers
└── results/
    ├── figures/                     # Generated publication-quality PNG charts
    │   ├── fig5_dsr_comparison.png
    │   ├── fig6_system_efficiency.png
    │   ├── fig7_execution_delay_robustness.png
    │   ├── fig8_pe_failure_robustness.png
    │   └── fig10_selfpilot_hardware.png
    └── *.json                       # Raw benchmark metrics datasets
```

---

## 🚀 Getting Started

### 1. Requirements
- Python 3.10+
- `numpy`, `scipy`, `pandas`, `matplotlib`, `plotly`, `streamlit`

### 2. Run All Unit Tests
Verify model validity, upward rank calculations, TOPSIS matrix normalization, scheduler allocation, and priority distribution curves:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Launch the Unified Interactive Web Dashboard
Run the dark-mode automotive HUD dashboard:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. Use the sidebar to seamlessly switch between:
1. **⚡ CASDS Dynamic DAG Scheduler**: Heterogeneous PEs (13 PEs for ADSuite, 11 PEs for SelfPilot), dynamic Gantt charts, paper figures 5-10, and fault tolerance labs.
2. **🏎️ Priority Assignment Strategies**: 4-Strategy Curves (Uniform, Arithmetic, Geometric, Logarithmic), Response Time Analysis (RTA), nuScenes perception topologies, and benchmark verification.
3. **🔗 Integrated Studio**: Live demonstration of how priority quantization curves shape CASDS TOPSIS scores on microcontroller register grids.

### 4. Run Paper Experiments Pipeline
- **Run CASDS Experiments**:
  ```bash
  python -m casds.experiments.run_all_experiments 60 2
  ```
- **Run Priority Scheduling Experiments**:
  ```bash
  python run_priority_experiments.py
  ```

---

## 📊 Benchmark Tables (Reproduced from Paper)

### Table I: Time Profile of Task Kernels on Each PE-Type (ADSuite Benchmark)
| Task Kernel | CPU [ms] | GPU [ms] | ASIC [ms] |
| :--- | :---: | :---: | :---: |
| **Object Detection (DET)** | 3531 | 156 | 96 |
| **Object Tracking (TRA)** | 1825 | 17 | 2 |
| **Localization (LOC)** | 165 | 95 | 10 |
| **Fusion (FUS)** | 0.1 | - | - |
| **Mission Planning (MiP)** | 1 | - | - |
| **Motion Planning (MoP)** | 8 | - | - |

### Table IV: Time Profile of SelfPilot Task Kernels on Real Hardware
| Task Kernel | CPU [ms] | GPU1 [ms] | GPU2 [ms] |
| :--- | :---: | :---: | :---: |
| **LiDAR Object Detection (LOD)** | 60 | 17 | - |
| **Camera Object Detection (COD)** | 62 | 25 | - |
| **Object Detection Fusion (ODF)** | 7 | - | - |
| **Lane Detection (LD)** | 22 | 9 | - |
| **Multiple Object Tracking (MOT)** | 29 | - | - |
| **Localization (LOC)** | 60 | 23 | 8 |
| **Mission Planning (MiP)** | 1 | - | - |
| **Motion Planning (MoP)** | 69 | - | - |
| **Control (CON)** | 11 | - | - |

---

## 📈 Key Findings & Experimental Insights

1. **DSR & Safety Criticality (Figure 5)**:
   - CASDS consistently achieves the highest $DSR(S)$ and $DSR(S_{crit})$ across all arrival rates ($\lambda = 20, 30, 40$ DAG/s) and critical ratios ($R_{crit} = 30\%, 60\%, 90\%$).
   - In congested scenarios ($\lambda = 40$ DAG/s, $R_{crit} = 90\%$), CASDS avoids the domino scheduling collapse suffered by ADS and ASDYS.
2. **System Efficiency (Figure 6)**:
   - CASDS achieves the lowest makespan and highest PE utilization, sustaining peak effective throughput $ET(S)$ across varying workloads.
3. **Robustness to Jitter and Faults (Figures 7 & 8)**:
   - Under execution time delay noise ($\mu_d \in [0.0, 1.0]$), CASDS's predictive PLCT re-estimates realistic deadlines on the fly.
   - Under transient PE failures ($\lambda_f \in [0, 10]$ failures/s), CASDS gracefully re-routes tasks without total system stall.
4. **ASIL Scalability (Table III)**:
   - With 4 criticality levels ($S_1, S_2, S_3, S_4$), CASDS achieves $100\%$ completion for safety-critical $S_4$ while preserving high completion rates for lower-tier tasks ($S_1, S_2$).
5. **Real Hardware Verification (Figure 10)**:
   - Evaluated on the TZTEK GEACX2 (dual Jetson AGX Orin) + Advantech MIC-75G30 platform running the 9-kernel SelfPilot autonomous driving pipeline.

---

## 📜 Citation

```bibtex
@article{zhang2026context,
  author={Zhang, Chuang and Zheng, Sifa and Chen, Chaoyi and Li, Haoran and Cheng, Hao and Xu, Qing and Li, Keqiang},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Context-Aware-Slack-Driven Dynamic Task Scheduling for Automotive Cyber-Physical Systems}, 
  year={2026},
  doi={10.1109/TII.2026.3704380}
}
```
