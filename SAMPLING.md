# Text2PDE Airfoil 多次采样

固定已经选中的动力学权重与配套 AE，在完整 Validation100 上每条轨迹独立生成 16 次。
K=1、2、4、8、16 共用同一采样池，各取前 K 次解码后的物理 UVP 求均值，再沿用原评价器评分。
统计同时保留单次误差、逐节点采样方差和各 K 的推理成本。训练种子固定，采样标签决定独立随机噪声。

任务为首帧预测未来 64 个存储帧，物理时间步为 0.0016。首帧保留观测值，未来所有节点的 UVP
由模型预测，翼面与入口标签沿用 Airfoil 原始定义。归一化复用 Train 统计，评价使用原网格和面积权重。
压力误差沿用原评价器的逐帧 gauge adjustment。Test 保持封存。

## 运行

在当前 `feature/airfoil-uvp-4gpu` 分支的独立 checkout 中，激活该方法原有 CUDA 环境。
使用已完成训练的权重和新的输出目录，在已有单进程、单 GPU 评价环境中执行：

```bash
export PYTHON=python
export NAS_ROOT=/path/to/nas/gladit
export RESULT_ROOT="$NAS_ROOT/runs/airfoil_text2pde_seed42"
export OUTPUT_DIR="$NAS_ROOT/results/airfoil_text2pde_seed42_sampling"
bash scripts/airfoil_sampling.sh
```

入口自动使用 [NAS 包装设置](NAS.md)。已有数据在其他位置时设置 `DATA_DIR`。
保留调度器的 GPU 分配，`DEVICE` 默认为 `cuda:0`。采样入口复用已准备数据和权重；
它遵循现有独立评价的 FP32、batch1 配方。四卡训练的配置与选优协议见 [Airfoil](AIRFOIL.md)。

入口默认读取该运行生成的 LDM 配置，以及 AE 和 LDM 的选优记录。每次生成使用原二十步 DDIM，联合生成未来六十四帧。配置中的数据位置和已有 Train normalizer 必须可访问。`CONFIG`、`CHECKPOINT`、`AE_CHECKPOINT` 可指定对应文件；AE 依赖身份按原入口核验。

可向命令追加 `--samples 8 --ensemble-sizes 1 2 4 8` 调整采样工作量，
或追加 `--skip-timing` 生成质量与方差结果，并在退出记录中标明测速待完成。
完整交付使用默认 16 次采样及全部五个 K。采样池大小至少为 2，且覆盖每个请求的 K。

## 统计与成本

每条轨迹以 float64 流式累计物理 UVP 均值和无偏方差，方差分母为样本数减一。
保存的场为 float32，均值场按相同保存精度评分。方差单位为各物理量的单位平方；
压力方差使用原始物理压力。空间方差汇总采用节点面积权重，未来 64 帧等权，
再对 100 条轨迹等权平均。首帧采样方差为零。

单次误差先在轨迹内平均，再对轨迹等权汇总。每个采样标签还分别给出 Validation100 分数，
并计算标签间标准差。逐轨迹采样标准差、跨轨迹差异与训练种子差异分别解释。
均值场误差通过评分平均后的流场得到。

质量评价之后，调用原共同 benchmark 对各 K 分别重新生成和计时。
使用固定 Validation24，每条轨迹两次预热、三次计时。计时涵盖 CPU 首帧输入、编码、
完整预测、解码、返回 CPU 和 float64 均值归约，也包含成员随机种子设置。
数据读取、评分及文件保存位于计时区间外。各次计时使用相同成员标签。

默认质量阶段生成 1,600 份预测；五个 K 的预热和计时再生成 3,720 份预测。
每个方法在独立获配时段运行，跨方法成本比较使用相同 GPU、环境和计时条件。

## 回传结果

- `summary.json`：单次误差、采样波动、逐轨迹统计和各 K 的完整物理指标。
- `ensemble.csv`、`members.jsonl`：K 曲线与逐轨迹、逐采样标签的指标和实际随机种子。
- `timing/`：各 K 的逐条耗时、显存、环境和计时定义。
- `manifest.json`、`exit.json`：数据、配置、权重、采样协议和退出状态。
- `fields/`：各 K 的物理均值场、网格、真值和最终逐节点方差。均值场采用原预测档案格式，可用于后续评分或渲染。

输出目录须为新目录，程序会拒绝覆盖已有结果。异常会保留已生成文件、失败轨迹或阶段及退出记录。
成功结束要求完整覆盖 Validation100，且输入文件身份保持一致。

## 验证状态

交付前执行 Python/Bash 语法、源码引用、Ruff 和 Git diff 静态检查。
当前 Windows 环境与接收端 CUDA 生产环境的等价性尚未核验，GPU 运行结果待接收端生成。
