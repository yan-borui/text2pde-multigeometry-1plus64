# NAS 运行入口

激活该方法原有 CUDA 环境，从当前仓库根目录执行。原子目录锁保护共享准备、缓存和训练目录，支持缺少 `flock` 命令或文件锁不可用的 NAS。

```bash
export PYTHON=python
# 替换为所有相关节点都能读写的 NAS 绝对路径。
export NAS_ROOT=/path/to/nas/gladit
```

`scripts/nas.sh` 先设置 `HDF5_USE_FILE_LOCKING=BEST_EFFORT`，再调用原入口。这会在支持的 HDF5 版本中保留可用的文件锁，并容忍文件系统禁用锁。已设置的环境变量优先。四卡、32 卡、单卡评价分别沿用各自原协议。

若 HDF5 仍报 `No locks available` 或 `Operation not supported`，可在原命令前设置 `HDF5_USE_FILE_LOCKING=FALSE`。此时每个输出文件应保持单写者，训练读取已经完整发布的 HDF5；共享准备仍使用本次目录锁。设置须在 Python 启动前生效。版本支持和读写约束见 [HDF5 官方文件锁说明](https://support.hdfgroup.org/documentation/hdf5/latest/_file_lock.html)。

## 启动

`NAS_ROOT` 下默认使用 `data/airfoil_raw` 和 `data/airfoil_uvp_stride8`。复用现有数据时显式设置 `RAW_DATA_DIR` 和 `DATA_DIR`。六仓库应指向同一份数据。保留调度器的 `CUDA_VISIBLE_DEVICES`；独立节点填写已经获配的四张卡。

```bash
export RESULT_ROOT="$NAS_ROOT/runs/airfoil_text2pde_seed0"
bash scripts/nas.sh bash scripts/airfoil_4gpu.sh train
# 当前修复版本所建运行的断点恢复：
bash scripts/nas.sh bash scripts/airfoil_4gpu.sh resume
```

首次启动包含完整数据和方法缓存准备。提前准备可把 `train` 换为 `prepare`。训练配方、环境与结果说明见 [Airfoil](AIRFOIL.md)。

## 共享目录与恢复

- 全部参与同一准备目录的进程使用本修复版本；旧版与新版各自采用不同锁协议。切换共享准备入口前，确认原准备任务已经退出。已有训练继续使用自己的冻结源码。
- 锁目录是原锁文件名加 `.d`，其中 `owner.json` 记录主机、PID和占用身份。等待表示另一个进程占用；训练目录的重复启动会立即报出锁的位置。
- 正常退出自动释放。中断或机器故障可能留下锁目录。核实记录主机上的进程及其所有子进程均已退出后，将该锁目录移至旁边的独立归档目录，再重试原入口。等待时长和其他主机上的同号 PID 都不能证明锁已失效。
- 每个训练任务使用独立结果目录。修复会改变源码身份；既有运行的严格源码校验继续保留。新代码使用新结果目录，新代码所建运行通过原 `resume` 参数恢复。

## 验证状态

已完成交付前 Python/Bash 语法和源码静态检查。NAS 的实际挂载语义、CUDA依赖及正式GPU运行仍由目标环境的原入口验证。当前机器为 Windows，缺少经核验等价的四卡/32卡生产环境。Test 继续封存。
