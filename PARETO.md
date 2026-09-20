# TEXT2PDE 同机推理测速与Validation100

使用既有已选checkpoint和当前正式配置，不重新训练。先分配与其他方法相同的单张GPU，
复用已核对的CUDA环境。记录相同CPU/GPU、Python、Torch、NumPy、CUDA/cuDNN版本。

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export CAMPAIGN=cylinderflow_same_machine_20260920
python pareto_run.py --config "$CONFIG" --checkpoint "$CHECKPOINT" --ae-checkpoint "$AE" \
  --campaign-id "$CAMPAIGN" --device cuda:0 --threads 2 \
  --output-dir "/shared/$CAMPAIGN/text2pde"
```

先将CHECKPOINT、DATA_DIR、PREPARED及所需AE/CONFIG设为本机现有绝对路径。
Text2PDE的CONFIG使用正式LDM YAML，数据目录沿用其中设置。AROMA必须使用动力学checkpoint
绑定的AE，Text2PDE必须使用LDM绑定的AE。保留原推理采样设置。

入口复用已有benchmark（固定Validation24、每条预热2次/测量3次），随后重新运行正式
Validation100；MGN/EAGLE每条一个样本，AROMA/Text2PDE每条三个独立单样本分数均值。
三个评分样本不会合成平均场，baseline图上只有一个点。耗时是单次完整64帧预测。
计时包括CPU输入处理、传输、模型及AE、解码与返回CPU，加载/评分/保存单列。

输出timing/、quality/、point.json及exit.json，保留原始计时、质量指标和预测。
将整个输出目录交给DiT仓库的`python -m graph_dit.pareto_plot`，用同一campaign ID
汇总绘图。不同环境和旧结果不会自动混入；具体命令见DiT仓库PARETO.md。
首次交付只做静态检查；GPU运行需在学长最终环境完成，Test封存。


统一交接与绘图：[DiT PARETO.md](https://github.com/yan-borui/graph-dit-cylinderflow-stride8/blob/main/PARETO.md)。
