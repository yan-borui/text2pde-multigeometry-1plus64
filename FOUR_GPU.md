# Text2PDE：四卡 AE → LDM（2026-09-12）

四卡入口保持两个阶段各 250,000 optimizer updates、global batch=4、seed42 和原
16-mixed 精度。单卡的 microbatch1 × accumulation4 改为四卡各 microbatch1 ×
accumulation1。AE 保留 1e-5 原生 cosine，LDM 保留 1e-4 和原生调度；H1 的
1e-6 下限不应用到 Text2PDE。原 YAML 模板继续保留单卡入口。

Train 1,000 条、Validation 100 条，均使用已发布 stride-8、Train75 归一化。
AE 读取每条全部 75 帧；LDM 固定读取前 65 帧，联合预测未来 64 帧（5.12 秒）。
训练诊断保留原先前 24 个 Validation 样例，四卡各六个；物理选优继续用原来的
均匀 Validation-24。最终选定 LDM 在完整 Validation-100 × seeds0/1/2 上评价，
Test 保持封存。

## 配置与启动

先按 [README.md](README.md) 准备官方数据、manifest 和 Train normalizer，使用已建好
的原模型运行环境。下面命令在仓库根目录执行，四卡须位于同一节点。

```bash
python -m tools.cylinderflow_stride8.run_four_gpu \
  --data "$DATA" --manifest "$MANIFEST" --normalizer "$NORMALIZER" \
  --result-root "$RESULT_ROOT" --gpus "$GPU_IDS"
```

`GPU_IDS` 为获配的四个 ID/UUID。该入口生成四卡配置后，依次完成：

1. AE 250k updates，保存四个原生里程碑 62,500/125,000/187,500/250,000。
2. 四卡分别评价四个 AE checkpoint，以原归一化 UVP L1、较早更新选 AE，冻结身份。
3. LDM 250k updates，保存相同四个里程碑。
4. 四卡分别评价四个 LDM checkpoint，每个使用 Validation-24 × 三个种子，20-step DDIM。
5. 所选 checkpoint 的 Validation-100 × 三个种子按轨迹分到四卡，合并 300 段结果，
   渲染代表样例并写入 Validation 完成与权重锁定记录。

单个四卡 Slurm 作业从头保持该分配：

```bash
sbatch scripts/slurm_four_gpu.sh "$PYTHON" \
  --data "$DATA" --manifest "$MANIFEST" --normalizer "$NORMALIZER" \
  --result-root "$RESULT_ROOT"
```

脚本使用调度器的 GPU mask；分区、时限和账户按实际集群设置。准备一份配置而不训练时：

```bash
python -m tools.cylinderflow_stride8.materialize_config \
  --template configs/cylinderflow_stride8/ae_1plus64.yaml --stage ae --world-size 4 \
  --data "$DATA" --manifest "$MANIFEST" --normalizer "$NORMALIZER" \
  --result-root "$RESULT_ROOT" --output "$AE_CONFIG"
```

## 有限步验收

分别使用新目录检查 AE 与 LDM 的真实四卡更新：

```bash
python -m tools.cylinderflow_stride8.run_four_gpu \
  --data "$DATA" --manifest "$MANIFEST" --normalizer "$NORMALIZER" \
  --result-root runs/preflight_ae_4gpu_new --gpus "$GPU_IDS" --preflight-stage ae

python -m tools.cylinderflow_stride8.run_four_gpu \
  --data "$DATA" --manifest "$MANIFEST" --normalizer "$NORMALIZER" \
  --result-root runs/preflight_ldm_4gpu_new --gpus "$GPU_IDS" \
  --preflight-stage ldm --ae-checkpoint "$SELECTED_AE"
```

每次上限为八次更新，验证参数发生有限变化、每卡状态与恢复点写出。
这两个入口只检查有界训练；完整 joint64 物理生成使用原评价入口或正式阶段结束后的
四卡 evaluator，不能把训练八步通过写成物理评价通过。

## 恢复和输出

相同命令加 `--resume` 使用 `checkpoints/last.ckpt` 恢复；已完成的阶段和评价候选复用。
数据按同一个 `seed + epoch` 排列后分片，rank 不重复读取窗口，记录全局样例数。
checkpoint 格式 `text2pde.cylinderflow.resume.ddp.v2` 包含 raw 权重、Adam/scheduler/
scaler、各卡 Python/NumPy/Torch/CUDA RNG、全局样例游标与配置/数据/AE 身份。
恢复拒绝 world size、配置、源码或 AE 依赖改变。四卡与旧单卡恢复格式分别处理。

源码冻结在各阶段 run 的 `source/`。Lightning 保存使用所有 rank 参与的回调，
公共文件由 rank0 写入；各卡失败日志独立。验证后恢复训练随机状态，训练最终记录
核对实际步数和窗口数，再允许阶段进入完成状态。评估逐样例提交完整结果，未完成
部分在故障恢复时补齐；失败渲染目录保留，重试生成新成品。

若故障发生在已保存预算终点之后，恢复直接交付已核验的训练状态，再补齐独立物理
选择与最终 Validation。该恢复记录明确标注训练内诊断没有重放；不会额外更新模型，
也不会把尚未完成的独立物理评价标为完成。

主要入口/记录：

- `config/`、`logs/`：实际四卡配置、各次命令、日志和退出码。
- `ae/formal/`、`ldm/formal/`：checkpoint、CSV loss/LR、各卡 `distributed_metrics.jsonl`。
- `training_completion.json`：实际更新数和全局样例数。
- `ae/selection_v1/`、`evaluation/ldm_selection_v1/`：物理选择结果及 checkpoint 身份。
- `evaluation/validation_v1/`：300 段预测与物理结果、代表动图。
- `evaluation/validation_complete_no_test_entry.json`：最终 AE/LDM 锁定及 Test 封存记录。

物理选择沿用原失败样例优先级、UV 指标、较早更新的排序规则；不新增早停。
`case_metrics_journal.jsonl` 保存逐次写入的过程，`completed_cases.json` 和最终
`case_metrics.jsonl`/CSV 为去重后的已完成样例集合。

## 本地验证范围

Torch-only 四卡分片/中途游标恢复两个检查通过；既有 protocol 六项、最终权重身份
一项、物理 evaluator 两项检查通过。新增真实 Lightning 双进程测试覆盖中途恢复跨
epoch、Adam/权重/各卡 RNG 精确一致，以及插入 Validation 不改变训练：

```bash
python -m unittest discover -s tests -p 'test_distributed_training_resume.py'
```

本机缺少 Lightning，该项已显式 **SKIP**，需要目标环境运行；未将其记为通过。
采样器针对 [Lightning 2.3.2 epoch-loop 的恢复计数](https://github.com/Lightning-AI/pytorch-lightning/blob/2.3.2/src/lightning/pytorch/loops/training_epoch_loop.py)
保留完整 epoch 长度，避免已消费计数和剩余长度叠加后提前截断。真实 Lightning
hook 顺序、混合精度、NCCL、模型显存和物理推理仍需目标环境验收。
本次源码交付没有启动正式训练。
