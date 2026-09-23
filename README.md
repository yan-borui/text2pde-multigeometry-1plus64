# Text2PDE · Airfoil UVP

**NAS 运行：**[启动入口与共享目录设置](NAS.md)。

`feature/airfoil-uvp-4gpu`提供默认四卡的Airfoil训练与恢复入口。
完整数据契约、环境、固定配方和命令见[Airfoil操作说明](AIRFOIL.md)。

沿用本方法既有四卡训练配方，适配官方Airfoil的UVP、时间轴和动态边界。

设置原始数据目录和四卡训练结果目录后，一条命令自动完成缺文件下载、
全量stride8转换、Train归一化、模型缓存准备及训练：

```bash
export RAW_DATA_DIR=/data/datasets/meshgraphnets/airfoil
export RESULT_ROOT=/shared/runs/airfoil_text2pde_seed42
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用Slurm时保留调度器分配的值
bash scripts/airfoil_4gpu.sh train
```

已有数据/缓存自动复用，`resume`恢复训练。只处理全量数据时运行
`bash scripts/airfoil_4gpu.sh data`，无需GPU。

此分支已做静态检查，目标四卡运行验收尚未执行。Test保持封存。
原上游源码、许可证和历史CylinderFlow记录保留。
