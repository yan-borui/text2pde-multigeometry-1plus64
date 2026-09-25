# Text2PDE Airfoil 采样分组统计

固定选中的动力学权重、AE、Train 归一化和完整 Validation100，重复运行 R 个独立采样组。
每组内将 K 次预测解码到物理 UVP，逐节点、逐时刻求均值，再对均值场运行完整评价。
每个 K 得到 R 个 Validation100 分数，报告分数均值、无偏方差与标准差。

任务为首帧预测未来 64 个存储帧，物理时间步为 0.0016。首帧保留观测值，
未来所有节点 UVP 由模型预测。原网格、面积权重、Airfoil 节点标签和压力 gauge adjustment
沿用既有评价器。Test 保持封存。

## 统计定义

对轨迹 $i$、组 $r$、ensemble 大小 $K$，先计算物理场均值：

$$
\bar{u}_{r,K,i}=\frac{1}{K}\sum_{j=1}^{K}u_{r,j,i}.
$$

原评价器对未来 64 帧计算面积加权 UV relative RMSE，再对 100 条轨迹等权平均：

$$
q_{r,K}=\frac{1}{100}\sum_{i=1}^{100}\operatorname{UVRelRMSE}
\left(\bar{u}_{r,K,i},u_i^{\mathrm{ref}}\right).
$$

跨组统计的样本单位为一次完整 Validation100 评价：

$$
\bar{q}_{K}=\frac{1}{R}\sum_{r=1}^{R}q_{r,K},\qquad
s_K^2=\frac{1}{R-1}\sum_{r=1}^{R}(q_{r,K}-\bar{q}_K)^2,
\qquad s_K=\sqrt{s_K^2}.
$$

同组不同 K 取共享采样池的前 K 个成员，使 K 曲线可以配对比较；组间使用互不重叠的成员。
训练种子与权重固定，这些统计描述采样波动。逐节点物理场方差保留为另一项场分布诊断。
图中的误差条为跨组标准差。

## 运行

在独立、已提交且 tracked 文件干净的 checkout 中激活本方法的既有 Linux CUDA 环境。
使用与正式独立评价一致的单进程、单 GPU、FP32、batch1 设置。
四卡训练的配置和选优协议见 [Airfoil](AIRFOIL.md)，环境包装见 [NAS](NAS.md)。

以下示例显式选择 R=5、K=1/2/4/8/16。R 与 K 根据资源决定；R 至少为 2。
已有 matplotlib 时追加 `--plot`，生成均值与方差曲线。

```bash
export PYTHON=python
export NAS_ROOT=/path/to/nas/gladit
export RESULT_ROOT="$NAS_ROOT/runs/airfoil_text2pde_seed42"
export OUTPUT_DIR="$NAS_ROOT/results/airfoil_text2pde_seed42_sampling_groups"
bash scripts/airfoil_sampling_groups.sh \
    --groups 5 --samples 16 --ensemble-sizes 1 2 4 8 16 --plot
```

入口读取 LDM 配置及 AE、LDM 的选优记录。每次生成采用二十步 DDIM，联合预测未来六十四帧。配置中的数据与已有 Train normalizer 必须可访问。`CONFIG`、`CHECKPOINT`、`AE_CHECKPOINT` 可指定对应文件；AE 依赖身份沿用原入口核验。

`DATA_DIR` 可指定已有数据，`DEVICE` 默认为 `cuda:0`。保留调度器的 GPU 分配。
`--samples` 控制每组采样池大小，至少为 2 且覆盖最大 K；`--seed-base` 控制起始采样标签。
每组、每轨迹、每成员的实际 PRNG seed 全部保存，并检查本次任务的 seed 互异性。
所有组的 case 顺序、权重、AE、归一化、数据和环境必须一致。

## 成本与恢复

质量阶段生成 `100 × R × samples` 个完整预测。每组依次复用原生评价入口，
以 float64 累计物理均值，并按保存的 float32 精度评分和保存。
各 K 的完整 benchmark 在第一组执行一次，后续组用于质量统计。
使用固定 Validation24，每条轨迹两次预热、三次计时。计时包含成员 seed 设置、
首帧输入、编码、预测、解码、返回 CPU 和物理均值归约；数据读取、评分、保存位于计时区间外。
K=1/2/4/8/16 的测速共生成 3,720 个额外预测。

资源优先用于质量统计时，首次运行追加 `--skip-timing`，输出会记录测速状态。
完成组作为恢复单位；保留相同命令和输出目录并追加 `--resume` 即可继续。
恢复会重新核验已完成组的身份、完整 case/member 清单和保存场文件，并复用其分数。
中断组保留全部日志与部分结果，在新的 attempt 目录使用原 seed 重跑。
同一输出目录使用进程锁，系统退出后自动释放。

恢复要求 R、K、采样池大小、seed、数据、配置、权重、源码提交和执行参数保持一致。
绘图开关可在恢复时补加。所有组完整时，`--resume --plot` 直接从已有分数生成图。
更换权重、AE、配置或采样计划时使用新输出目录。

## 回传结果

- 根目录 `summary.json`、`score_curve.csv`：各 K 的 R 个分数、均值、无偏方差、标准差。
- `group_scores.csv`：每组、每 K 的完整 Validation100 分数。
- `score_curve.pdf/png`：请求绘图时生成，左图为均值及标准差，右图为分数方差。
- `manifest.json`、`exit.json`：输入文件身份、源码提交、配置、seed 协议和完成状态。
- `groups/`：每组原始评价结果、成员分数、所有 K 的均值场、逐节点方差、
  场指标、运行日志与退出码。第一组还包含各 K 的成本测量。

`sampling_ensemble.py` 与 `scripts/airfoil_sampling.sh` 保留单组入口。
独立组统计通过 `sampling_groups.py` 与 `scripts/airfoil_sampling_groups.sh` 运行。

## 验证状态

实现交付采用源码、Python/Bash 语法、Ruff 和 Git diff 静态检查。
当前 Windows 环境与正式 Linux CUDA 环境的等价性尚未核验，运行验证与数值结果待目标环境生成。
