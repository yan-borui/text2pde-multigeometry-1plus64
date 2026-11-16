# Text2PDE 长时滑动窗口预测

固定已有选中权重，以真实第 0 帧为唯一观测条件，生成未来 256 帧（dt=0.08，总计 20.48 秒）。默认 64 帧窗口、步长 32、K=8 物理场均值、FP32、单 CUDA GPU，与 GLaDiT 的长时窗口设置一致。

## 生成语义

窗口起点为 0、32、64、96、128、160、192。各窗口的 K 个独立成员先按 FP32 求物理 UVP 均值，不同窗口对同一未来帧再等权平均。下个窗口以起点已经融合的物理场为条件；条件帧不额外参与平均。入口/壁面标签 4/6 的 UV 始终回填原始第 0 帧，压力保留预测值。

沿用原配置及其 Train 归一化，校验 LDM/AE 绑定。每个窗口使用原 `sample_initial64`，保留 20 步 DDIM，联合生成未来 64 帧。物理场条件重新编码；解码器空间坐标按原轨迹范围归一化，局部时间每窗重置为 0–1。

种子为 `(derive_sample_seed(member + label*K, trajectory_index) + start*104729) % (2**31-1)`。首窗口 label0 的前 3 个成员对应原评测标签 0/1/2。默认每条轨迹 56 次联合采样、1,120 次去噪网络调用。

这是独立的长时推理入口。K=8 是本入口的窗口均值设置；原三次独立采样、分别评分的 Validation/Test 入口及训练流程保留。可用 `--ensemble-size 1 --sampling-label 0` 生成单成员轨迹；不同 K、窗口重叠及方法对应不同计算成本。

## 运行

先使用所选权重对应的既有 CUDA 环境和数据依赖，在仓库根目录运行。输出目录必须不存在，避免覆盖已有结果。

```bash
python -m tools.cylinderflow_stride8.sliding_window \
  --config /absolute/path/to/selected_ldm_config.yaml \
  --checkpoint /absolute/path/to/selected_ldm.ckpt \
  --ae-checkpoint /absolute/path/to/matching_ae.ckpt \
  --scope validation --case-count 3 \
  --future-frames 256 --stride 32 --ensemble-size 8 \
  --device cuda:0 --output-dir /absolute/path/to/new_sliding_result
```

默认按 split 顺序取前 3 条；`--indices 1000 1001 1002` 显式指定 Validation 全局编号。Test 需先按 [TEST.md](TEST.md) 准备与原训练统计绑定的 Test65，再用 `--scope test --test-data-dir /absolute/path/to/test65`；Test 编号为 0–99，默认 0、1、2。保留全部指定样本和失败，不按误差筛选案例或随机种子。

未来真值只在全部窗口生成完成后读取。指标仅计算前 64 个未来帧，分别保存首窗口和融合结果；融合结果使用额外窗口计算。第 65 个未来帧起没有真值误差分数。参考数据仍限制为原 1+64 前缀。

## 输出与验证

- `request.json`、`identity.json`、`source.zip`：请求、权重/AE/归一化绑定、数据与配置、采样规则、实际 GPU 与推理源码。
- `trajectory_*/window_*.npz`、`windows.json`：每个窗口的物理预测、条件、种子、全局帧号及耗时；`--save-members` 另存每个成员。
- `rollout.npz`：融合轨迹、参考前缀、网格、时间轴与贡献次数。
- `summary.json`、`failure.json`：逐案例指标、耗时、相邻帧 UV 变化及异常。存在失败则非零退出，保留已完成窗口。
- `rollout.gif`、`frame_*.png`：速度、压力、涡量共享色标图。色标取参考前缀，超出参考时段的真值面板留空；`--no-render` 省略绘图。

2026-11-16：完成静态语法、名称和差异检查，以及原生接口、条件帧、边界、窗口索引和参考读取顺序核查。未在匹配的生产 CUDA 环境运行，尚无本轮长时预测结果。
