# VecFlow 集成实现总结

时间：2026-04-16

本文档对应 [INTEGRATION_PLAN.md](/home/lzg/VecFlow-develop/INTEGRATION_PLAN.md)，说明当前代码里每个主要点是如何实现的、代码在哪里、哪些点已经完成，哪些点按当前版本边界做了收敛。

## 1. 总体结论

当前已经完成并可用的主链路：

- VecFlow C++ 单卡接口
- Tagore 构图接入
- Phoenix C++ 接口接入
- Phoenix 分层存储与按标签加载
- 多 GPU C++ API 与 bench coordinator
- label-aware routing
- `storage_stats(...)` 统计接口

当前明确收敛后的边界：

- 不再保留 `Tagore native` 搜索路径
- Tagore 统一采用“构图由 Tagore 完成，搜索统一复用 cuVS `cagra::filtered_search`”

这意味着当前实现追求的是：

- C++ 主链路稳定可用
- Phoenix / 多 GPU / Tagore 能端到端工作
- 性能目标在主路径上达标

而不是继续维护一条额外的 native 搜索分支。

## 2. 设计方案

### 2.1 图构建与搜索分工

当前设计：

- `CAGRA` builder：
  - 每个高频 label 直接用 cuVS CAGRA 构图
- `TAGORE_CAGRA_COMPAT` builder：
  - 每个高频 label 用 Tagore 构图
  - 输出改写成 CAGRA-compatible 图布局
- 搜索：
  - 无论图来自 CAGRA 还是 Tagore，只要是 graph 路径，都统一调用 cuVS `cagra::filtered_search`

这样做的原因：

- 只保留一条搜索实现，接口和缓存格式更稳定
- Phoenix label-load / 多 GPU / benchmark 都可以复用同一套搜索逻辑
- 避免 native kernel、entry-point cache、特殊图布局带来的额外维护成本

相关代码：

- [cpp/include/cuvs/neighbors/vecflow.hpp](/home/lzg/VecFlow-develop/cpp/include/cuvs/neighbors/vecflow.hpp)
- [cpp/src/neighbors/vecflow/vecflow_build.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_build.cuh)
- [cpp/src/neighbors/vecflow/vecflow_search.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_search.cuh)
- [cpp/src/neighbors/vecflow/tagore_build.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/tagore_build.cuh)

### 2.2 Phoenix 分层存储设计

当前实现将 graph 和 packed dataset 分开管理，但都采用相同的三层思路：

- HBM：
  - 当前最热 label 的 graph rows
  - 当前最热 label 的 packed dataset rows
- DRAM：
  - 被 HBM 淘汰但仍有复用价值的 label 数据
  - 使用 pinned host memory，便于更快重新提升到 HBM
- SSD：
  - 完整 cache 文件保存在磁盘
  - graph 通过 Phoenix 读取
  - packed dataset 也按 label 切片读取

具体到数据放置：

- graph：
  - CAGRA-compatible label-local rows
- dataset：
  - 按 `cagra_index_map` 打包后的 label-local向量块
  - 这样 Phoenix label-load 时不需要保留完整 dataset 常驻 GPU

支持的行为：

- 初始放置：
  - build/load 后按容量先尽量放 HBM，再放 DRAM，剩余留在 SSD
- 按需加载：
  - 搜索某个 label 时，如 HBM 未命中，则从 DRAM 或 SSD 提升
- 预取：
  - 当前 label 搜索时，可为未来一个 label 预取 graph / dataset
- 动态重平衡：
  - 周期性使用 `access_count / bytes` 评分做升降级

相关代码：

- [cpp/src/neighbors/vecflow/phoenix_graph_load.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/phoenix_graph_load.cuh)
- [cpp/src/neighbors/vecflow/vecflow_build.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_build.cuh)
- [cpp/src/neighbors/vecflow/vecflow_search.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_search.cuh)
- [cpp/include/cuvs/neighbors/vecflow.hpp](/home/lzg/VecFlow-develop/cpp/include/cuvs/neighbors/vecflow.hpp)

### 2.3 多 GPU 设计

当前多 GPU 分两层：

- 库内 C++ API：
  - `build_multi_gpu(...)`
  - `search_multi_gpu(...)`
- bench coordinator：
  - `VECFLOW_MG_BENCH`
  - 多进程方式为每张卡拉起一个 worker

查询分发支持两种模式：

- contiguous query slicing
- label-aware routing

label-aware routing 的策略：

- 先统计每个 label 的 query 数和 data 数
- 用 weighted bin-packing 将 label 分给不同 GPU
- 每个 worker 只处理自己负责的 label 查询，并在需要时只保留自己负责的 label 子集

相关代码：

- [cpp/src/neighbors/vecflow/multi_gpu.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/multi_gpu.cuh)
- [cpp/src/neighbors/vecflow/vecflow_build.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_build.cuh)
- [cpp/src/neighbors/vecflow/vecflow_search.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_search.cuh)
- [vecflow/examples/cpp/src/bench/vecflow_mg_bench.cpp](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/vecflow_mg_bench.cpp)

## 3. 计划项对照

### 3.1 Tagore 集成

状态：已完成，但收敛为 compat-only

实现：

- Tagore builder 已接入
- 支持 192D 编译运行
- 输出 compat 图布局
- 搜索统一走 cuVS

代码：

- [cpp/src/neighbors/vecflow/tagore_build.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/tagore_build.cuh)
- [cpp/src/neighbors/vecflow/vecflow_build.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_build.cuh)

说明：

- 原计划里 native search kernel 这部分，当前版本不继续保留，已经删除。

### 3.2 Phoenix 接入

状态：已完成

实现：

- Phoenix graph load
- Phoenix label load
- graph / dataset 双缓存
- HBM / DRAM / SSD 三层
- 预取
- 动态重平衡

代码：

- [cpp/src/neighbors/vecflow/phoenix_graph_load.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/phoenix_graph_load.cuh)
- [cpp/src/neighbors/vecflow/vecflow_build.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_build.cuh)
- [cpp/src/neighbors/vecflow/vecflow_search.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_search.cuh)

### 3.3 多 GPU 调度

状态：已完成

实现：

- 库内 `build_multi_gpu` / `search_multi_gpu`
- bench coordinator
- label-aware routing
- weighted load 统计
- `storage_stats(...)` 能返回 worker ownership / load / storage tier

代码：

- [cpp/include/cuvs/neighbors/vecflow.hpp](/home/lzg/VecFlow-develop/cpp/include/cuvs/neighbors/vecflow.hpp)
- [cpp/src/neighbors/vecflow/multi_gpu.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/multi_gpu.cuh)
- [vecflow/examples/cpp/src/bench/vecflow_mg_bench.cpp](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/vecflow_mg_bench.cpp)

### 3.4 实验与 smoke

状态：已完成

实现：

- 单卡 benchmark
- 多 GPU benchmark coordinator
- Tagore smoke
- 多 GPU API smoke
- Phoenix smoke

代码：

- [vecflow/examples/cpp/src/bench/vecflow_bench.cu](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/vecflow_bench.cu)
- [vecflow/examples/cpp/src/bench/vecflow_mg_bench.cpp](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/vecflow_mg_bench.cpp)
- [vecflow/examples/cpp/src/bench/vecflow_tagore_smoke.cu](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/vecflow_tagore_smoke.cu)
- [vecflow/examples/cpp/src/bench/vecflow_mg_api_smoke.cu](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/vecflow_mg_api_smoke.cu)
- [vecflow/examples/cpp/src/bench/vecflow_phoenix_smoke.cpp](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/vecflow_phoenix_smoke.cpp)

## 4. 关键接口

### 4.1 单卡

接口：

- `cuvs::neighbors::vecflow::build(...)`
- `cuvs::neighbors::vecflow::search(...)`
- `cuvs::neighbors::vecflow::storage_stats(index)`

代码：

- [cpp/include/cuvs/neighbors/vecflow.hpp](/home/lzg/VecFlow-develop/cpp/include/cuvs/neighbors/vecflow.hpp)

### 4.2 多卡

接口：

- `cuvs::neighbors::vecflow::build_multi_gpu(...)`
- `cuvs::neighbors::vecflow::search_multi_gpu(...)`
- `cuvs::neighbors::vecflow::storage_stats(multi_gpu_index)`

代码：

- [cpp/include/cuvs/neighbors/vecflow.hpp](/home/lzg/VecFlow-develop/cpp/include/cuvs/neighbors/vecflow.hpp)

## 5. 主要代码位置索引

- 公共 API：
  - [cpp/include/cuvs/neighbors/vecflow.hpp](/home/lzg/VecFlow-develop/cpp/include/cuvs/neighbors/vecflow.hpp)
- 构建主流程：
  - [cpp/src/neighbors/vecflow/vecflow_build.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_build.cuh)
- 搜索主流程：
  - [cpp/src/neighbors/vecflow/vecflow_search.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_search.cuh)
- 查询分类 / scratch：
  - [cpp/src/neighbors/vecflow/vecflow_common.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/vecflow_common.cuh)
- Tagore builder：
  - [cpp/src/neighbors/vecflow/tagore_build.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/tagore_build.cuh)
- Phoenix：
  - [cpp/src/neighbors/vecflow/phoenix_graph_load.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/phoenix_graph_load.cuh)
- 多 GPU：
  - [cpp/src/neighbors/vecflow/multi_gpu.cuh](/home/lzg/VecFlow-develop/cpp/src/neighbors/vecflow/multi_gpu.cuh)
- bench / smoke：
  - [vecflow/examples/cpp/src/bench/vecflow_bench.cu](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/vecflow_bench.cu)
  - [vecflow/examples/cpp/src/bench/vecflow_mg_bench.cpp](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/vecflow_mg_bench.cpp)
  - [vecflow/examples/cpp/src/bench/README.md](/home/lzg/VecFlow-develop/vecflow/examples/cpp/src/bench/README.md)

## 6. 当前版本的明确边界

已完成：

- C++ 主链路
- Tagore compat
- Phoenix 分层存储
- 多 GPU
- yfcc10m 主路径性能达标

不再包含：

- Tagore native kernel
- native entry-point cache
- native 特殊图布局
- native benchmark / mg benchmark 入口

