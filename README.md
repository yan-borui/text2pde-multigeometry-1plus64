# Text2PDE · Airfoil UVP

`feature/airfoil-uvp-4gpu`提供默认四卡的Airfoil训练与恢复入口。
完整数据契约、环境、固定配方和命令见[Airfoil操作说明](AIRFOIL.md)。

沿用本方法既有四卡训练配方，适配官方Airfoil的UVP、时间轴和动态边界。

1. 从官方Train/Validation TFRecord生成共享75帧UVP数据。
2. 按[AIRFOIL.md](AIRFOIL.md)完成该方法的准备阶段。
3. 在已分配的四卡节点运行`bash scripts/airfoil_4gpu.sh train`；中断后使用`resume`。

此分支已做静态检查，目标四卡运行验收尚未执行。Test保持封存。
原上游源码、许可证和历史CylinderFlow记录保留。
