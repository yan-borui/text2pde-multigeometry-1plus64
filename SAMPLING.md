# Text2PDE 重复采样与物理均值场

固定已选动力学权重、配套AE和原采样器，对完整Validation100各生成16条独立预测。
每个K使用同一采样池的前K条，在反归一化及原边界处理后对物理UVP求均值，沿用原物理评价器。
默认K为1、2、4、8、16；训练种子保持checkpoint中的值，采样种子按轨迹与标签生成。
Test保持封存。本入口先用于CylinderFlow，Airfoil适配需沿其数据、时间步和边界语义接入。

## 运行

在本分支的独立checkout内，激活原CUDA环境，使用已有已选checkpoint与绑定的AE：

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python sampling_ensemble.py \
  --config "$CONFIG" --checkpoint "$CHECKPOINT" --ae-checkpoint "$AE_CHECKPOINT" \
  --output-dir "$RESULT_ROOT" --device cuda:0
```

CONFIG沿用选中运行的配置。确认其中的数据路径和normalizer指向已有正式CylinderFlow数据。
RESULT_ROOT使用全新目录。模型推理为原FP32单卡评价路径，batch1；重复次数由--samples控制，
--ensemble-sizes选择其前缀。减少这些参数仅改变采样工作量，完整交付使用默认值。

每条轨迹流式累计均值与方差，保留16次各自的全部标量指标、五个均值场和最终逐点UVP方差。
默认不保存16份成员场。方差使用物理单位平方、ddof=1；压力方差为原始压力，
压力误差指标继续使用原评价器的gauge adjustment。空间汇总使用节点面积权重，未来64帧等权。
固定轨迹内的采样标准差与跨轨迹差异分别保留。

质量评价之后，调用现有共同benchmark分别测量各K。每个K实际重新生成K份并平均，
固定Validation24、两次预热和三次计时。计时涵盖CPU初始场至CPU物理均值场，
包含编码、推理、解码、传输及均值归约；数据读取、评分和保存位于计时区间外。
--skip-timing仅生成质量与方差结果，退出记录明确标注测速尚待完成。
一次完整调用包含1,600份质量预测及3,720份预热/计时预测；请预留独立GPU时段。

## 回传

- summary.json：单次误差均值、各采样标签的Validation100分数及其标准差、逐轨迹采样统计、各K完整物理指标。
- ensemble.csv：E6折线图所需的K与UV误差。
- members.jsonl：每条轨迹每个采样标签的指标，便于重新汇总。
- timing/：每个K的逐条计时和显存统计。
- manifest.json、exit.json：权重、配置、环境及退出状态。
- fields/：均值场与逐点方差，保留在服务器；需要渲染时再传相应轨迹。

跨采样标签的分数标准差衡量固定模型的采样波动。它与各轨迹内部的采样标准差、
以及重新训练种子的波动有不同含义。E6先并列展示结果，再判断哪种模型更受益于平均。
失败会保留已生成文件和异常信息；修复后使用新的结果目录。

## 验证状态

交付前完成静态语法、引用、格式与diff检查。正式CUDA环境运行和结果尚待接收端完成。
复用当前已有依赖，不安装额外模型包。现有训练、选优和benchmark入口保持原协议。
