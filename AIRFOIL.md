# Airfoil UVP · 四卡训练

分支：`feature/airfoil-uvp-4gpu`。在一个节点上使用四张已分配的 CUDA GPU；
`scripts/airfoil_4gpu.sh` 是当前训练/恢复入口。原 CylinderFlow 文档及测试记录保留其历史适用范围。

## 自动完成全量数据准备

训练入口自动执行完整数据准备，不需要手工下载、转换或运行单独的准备步骤。
默认原始目录为`/data/datasets/meshgraphnets/airfoil`，可通过`RAW_DATA_DIR`修改。
`DATA_DIR`默认是原始目录加`_uvp_stride8`；六仓库指向同一个目录即可共享数据。

入口依次完成：

1. 复用已有`meta.json`、`train.tfrecord`、`valid.tfrecord`；缺失文件从官方地址自动下载，支持断点续传。
2. 流式转换全部Train1000条、Validation100条，时间stride8，固定前75帧，仅保留UVP。
3. 计算Train-only归一化，生成HDF5、manifest和`text2pde_normalizer.pkl`。
4. 自动生成该方法的图/统计量缓存；DiT在取得VGAE权重后自动生成Train latent缓存。
5. 通过既有四卡入口开始训练。已完成的数据和缓存自动复用；多个仓库同时启动会等待共享锁，避免重复转换。

若只处理数据而不启动模型：

```bash
RAW_DATA_DIR=/data/datasets/meshgraphnets/airfoil bash scripts/airfoil_4gpu.sh data
```

这一步只使用CPU和已有NumPy/h5py，不需要GPU或TensorFlow。离线环境可设置`AIRFOIL_OFFLINE=1`。
数据日志和状态分别为`$DATA_DIR/preparation.log`、`preparation_status.json`。
转换在独立暂存目录进行，全部1100条完成后再发布；失败的暂存数据和日志保留，重试入口会重新转换。
输出仍为`airfoil_stride8_75frames.h5`、`airfoil_stride8_75frames_manifest.json`和normalizer。
Test不下载、不读取。`prepare`可选地提前完成本方法缓存，`train`会自动包含这些步骤。

## 固定任务

- 原始601帧、raw dt=0.0002；取raw0,8,...,592，共75帧。raw600不加入这一固定窗口。
- stored dt=0.0016；表示学习可用stored0..74，动力学评价为frame0→1..64，跨度0.1024。
- 模型只使用u/v/p三通道。密度保留在原始数据中，不作为模型输入、标签或统计量。
- Train1000、Validation100，沿用原始split，global ID分别0..999和1000..1099。
  UVP统计来自全部Train75帧；初始来流速度大小统计来自每条Train的frame0。
- 官方标签NORMAL=0、AIRFOIL=2、INFLOW=4保留在数据和预测档案中。
  已观察到type2和type4的速度随时间变化，因此所有节点的未来UVP均由模型预测。
  输入只含首帧和静态网格，关闭CylinderFlow的首帧边界固定回填。
- 网格和UVP保留源单位，物理评价使用原网格；每个方法的归一化沿既有实现。
  归一化、checkpoint、缓存和预测格式均具有独立Airfoil身份，拒绝混用CylinderFlow表示。
- 动力学主指标仍为未来64帧、面积加权UV relative RMSE，三次采样先在轨迹内平均，
  再对固定Validation-24等权平均。压力、涡量、谱和翼面误差保留。Airfoil的散度量是
  可压缩流场诊断，不作为零散度约束或不可压缩性达标依据。

官方数据schema和标签说明见
[MeshGraphNets数据读取实现](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/dataset.py)及
[NodeType定义](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/common.py)。

## 环境与分配

沿用本仓库已有模型依赖和集群CUDA环境。所有命令从仓库根目录执行。
先激活环境，再设置`PYTHON`为其解释器；也可使用当前`python`。
GPU数量固定为4；调度器已设置`CUDA_VISIBLE_DEVICES`时保持原值。

```bash
export PYTHON=python
# 在独立获配的四卡节点上设置；使用Slurm时保留调度器的mask。
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

每个方法使用独立结果目录。恢复时保持数据、源码、配置、四卡拓扑和依赖版本一致。
启动器记录日志和退出码；已有结果通过`resume`继续。

## 训练配方与命令

沿用既有四卡AE→LDM方案：两个阶段各250,000 optimizer updates，四卡各B1、global B4，
seed42、16-mixed；原生GINO/DiT结构、loss、LR和日程保持。AE读取75帧，LDM读取首65帧。
每阶段四个里程碑62,500/125,000/187,500/250,000；沿既有规则选AE后冻结，再训练/选择LDM。

```bash
export RESULT_ROOT=/shared/runs/airfoil_text2pde_seed42
bash scripts/airfoil_4gpu.sh train
# 中断后：
bash scripts/airfoil_4gpu.sh resume
```

原始数据转换已输出Train-only `text2pde_normalizer.pkl`，不重新拟合Validation统计。
启动器生成四卡配置并执行AE、AE选优、LDM、LDM选优、最终Validation100×3评价。
数据模块与入口保留`cylinderflow_stride8`名称以复用Lightning流程；实际契约、节点、时间轴
和checkpoint阶段身份已切换到Airfoil。HDF5使用兼容布局，移除旧CylinderFlow文件的HDF5 2.0专用门槛。

## 本次验证范围

本次完成源码、配置、Python/JSON/TOML语法、shell `bash -n`、Ruff F/E9和Git空白检查。
未启动Airfoil模型训练或四卡验收；四卡目标设备、NCCL和最大网格容量仍需在实际集群环境确认。
全量数据转换属于CPU数据准备，其完成状态、日志及轨迹数记录在`$DATA_DIR`中。
历史CylinderFlow测试与结果不构成本分支的运行证据。5090上的既有训练源码和GPU进程保持原状。
