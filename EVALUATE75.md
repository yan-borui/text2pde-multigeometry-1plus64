# Text2PDE Validation75 / Test75 评测

以观测第 0 帧预测未来 74 帧，总计 75 帧，dt=0.08，覆盖 5.92 秒。默认完整 100 条轨迹，固定已有选中权重、单 CUDA GPU、FP32；运行使用匹配的生产环境。完整 74 帧误差为本入口的主结果，同时保留前 64 帧指标。

## Test75 数据

沿用官方下载器取得完整原始 Test，不需要重新下载已有文件：

```bash
bash scripts/download_test.sh --dataset cylinderflow --raw-root /shared/data/mgn-test-raw
python -m mgn_data.prepare_test \
  --dataset cylinderflow \
  --raw-dir /shared/data/mgn-test-raw/cylinder_flow \
  --train-manifest /absolute/path/to/original_training_manifest.json \
  --frames 75 --output-dir /absolute/path/to/new_test75
```

从官方 600 帧中保留 raw 0,8,…,592，构建独立完整 Test100，每条 `[75,N,3]`。归一化逐项复制该模型的 Train manifest，不在 Test 拟合。转换检查全部 100 条轨迹与帧数，失败保留临时目录。Test75 必须使用新目录；已有 Test65 不能作为 Test75 复用。转换器不加 `--frames` 时仍保留原 65 帧默认；75 帧选项仅适用于本次 CylinderFlow 协议。

Validation 使用原 Train/Validation75 数据中的官方 100 条 Validation 轨迹，不重新分割或改变统计。旧模型的训练曝光与选权依据保持原记录，75 帧入口使用固定选中权重。

## 两个 split 的运行命令

从仓库根目录执行，两个结果目录均须不存在：

```bash
python -m tools.cylinderflow_stride8.evaluate75 \
  --config /absolute/path/to/selected_ldm_config.yaml \
  --checkpoint /absolute/path/to/selected_ldm.ckpt \
  --ae-checkpoint /absolute/path/to/matching_ae.ckpt \
  --scope validation --device cuda:0 \
  --output-dir /absolute/path/to/new_validation75

python -m tools.cylinderflow_stride8.evaluate75 \
  --config /absolute/path/to/selected_ldm_config.yaml \
  --checkpoint /absolute/path/to/selected_ldm.ckpt \
  --ae-checkpoint /absolute/path/to/matching_ae.ckpt \
  --scope test --test-data-dir /absolute/path/to/new_test75 \
  --device cuda:0 --output-dir /absolute/path/to/new_test75_result
```


## 预测与评分口径

保留原生 20 步 DDIM 与三个独立采样标签 0/1/2，每个标签按 64 帧窗口、步长 32 续生成，最终保留 0–74 帧。窗口起点 0、32，后窗条件来自该标签的生成帧；仅重叠窗口预测做物理场平均。每个标签分别评分，再平均同轨迹指标、对轨迹等权汇总；本入口不使用 K8 成员均值。每轨迹三标签共 6 次窗口采样，120 次去噪调用，另计编解码。

完整生成结束后才读取该样本的未来参考。入口严格要求 Test75；Test65 不会被静默截断或补齐后计作 75 帧评测。保留原生入口/壁面 UV 处理；压力及其空间均值处理沿用现有指标。UV relative RMSE 将全部未来帧的面积加权平方误差除以真值的面积加权平方和，再开方；逐帧 UV 误差另外记录。UV RMSE 分母和频谱轴按 74 个未来帧计算。

默认 Validation 编号 1000–1099、Test 编号 0–99。诊断可用 `--case-count` 或 `--indices` 明确缩小范围；输出记录实际样本集合，子集结果不标为完整 100 条评测。所有指定样本及失败均保留，失败非零退出。

`--stride` 可显式改变窗口步长，默认 32。结果保留 `full74`、融合轨迹的 `prefix64` 和原首窗口 `first_window64`，后两项有不同的采样成本。原 `sliding_window` 的 256 帧可视化入口保持原默认。

## 输出和验证

`identity.json`、`request.json` 和源码快照记录权重、AE/表示、数据/统计、split、帧范围、采样方式及环境。`trajectory_*_seed_*/prediction.npz` 与 `reference.npz` 分别保存完整 75 帧预测和真实参考；`metrics.json` 保留逐样本及逐帧指标；`summary.json` 与 `*_trajectory_metrics.csv` 分别汇总不同评分长度。错误保存在 `failure.json`。逐窗口种子、条件与耗时另保存在窗口文件；可用 `--save-members` 保存成员。

2026-11-16：代码与说明完成静态检查；未核实生产等价 CUDA 环境，完整 Test75 转换与 GPU 评测未执行，暂无本轮数值结果。既有训练、模型选择和 65 帧评测默认保持兼容。
