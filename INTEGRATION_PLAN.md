# VecFlow + Phoenix + Tagore 整合计划

> 目标：在消费级显卡上实现过滤ANNS QPS的接近线性扩展，支持超出HBM容量的数据集。

## 0. 现状与目标

### 0.1 三个系统现状

| 系统 | 会议 | 核心能力 | 当前瓶颈 |
|------|------|----------|----------|
| VecFlow | SIGMOD'26 | 标签特异性 → IVF-CAGRA + IVF-BFS 双索引 | 所有数据必须在HBM |
| Phoenix | SC'25 | 重构GDS I/O栈，消费级显卡可用DMA | 仅是I/O栈，无ANNS语义 |
| Tagore | — | GNN-Descent + CFS剪枝，多GPU流水线 | 仅加速构建，无搜索内核 |

### 0.2 整合目标

1. **超HBM容量**：用Phoenix实现SSD-DRAM-HBM三级存储，数据集大小从HBM（24GB）扩展到SSD（TB级）
2. **Tagore图索引**：用Tagore替换cuVS CAGRA，作为高选择性标签的图索引（构建 + 图结构），支持多GPU并行per-label构建
3. **多GPU线性扩展**：N张消费级显卡 → QPS接近N倍增长，零跨GPU通信

### 0.3 核心设计原则

- **选择性（selectivity）决定索引类型**：高选择性 → Tagore图索引（替代原CAGRA），低选择性 → BFS索引
- **访问频率（access frequency）决定存储层级**：两者是正交维度，不能混淆
- **分级粒度是标签（label），不是向量（vector）**：同一向量在不同标签子图中可处于不同存储层

---

## 1. 架构总览

```
                    ┌─────────── 查询路由器（按标签 + 按GPU） ──────────┐
                    │                                                    │
         ┌──────────┴──────────┐                          ┌─────────────┴─────────┐
         │      GPU 0          │           ...            │         GPU N          │
         │  ┌───────────────┐  │                          │  ┌────────────────┐    │
         │  │ HBM 热标签池  │  │                          │  │ HBM 热标签池   │    │
         │  │ (Tagore/BFS)  │  │                          │  │ (Tagore/BFS)   │    │
         │  ├───────────────┤  │                          │  ├────────────────┤    │
         │  │ DRAM 温标签池 │  │                          │  │ DRAM 温标签池  │    │
         │  │ (pinned host) │  │                          │  │ (pinned host)  │    │
         │  ├───────────────┤  │                          │  ├────────────────┤    │
         │  │ SSD 冷标签    │  │                          │  │ SSD 冷标签     │    │
         │  │ (Phoenix DMA) │  │                          │  │ (Phoenix DMA)  │    │
         │  └───────────────┘  │                          │  └────────────────┘    │
         └─────────────────────┘                          └────────────────────────┘
```

### 1.1 双维分类策略

```
第一步（构建时，静态）：按选择性决定索引类型
    cat_freq[label] > specificity_threshold → Tagore图索引
    cat_freq[label] ≤ specificity_threshold → BFS索引

第二步（运行时，动态）：按访问频率决定存储层级
    value(L) = access_frequency(L) / data_size(L)
    value 排序 → 前 N% 在 HBM，中间在 DRAM，其余在 SSD
```

### 1.2 新的 index 结构（概念设计）

```cpp
// 新增：per-label 元数据
struct label_meta {
    uint32_t label_id;
    IndexType index_type;      // TAGORE_GRAPH | BFS （由选择性决定，构建时确定）
    StorageTier tier;           // HBM | DRAM | SSD （由访问频率决定，运行时动态）
    size_t data_size;           // 该标签子图 + 向量的总字节数
    uint64_t access_count;      // 累计查询命中次数
    uint64_t last_access_ts;    // 最近访问时间戳
    // 存储位置（三者互斥）
    void* hbm_ptr;              // 若在HBM
    void* dram_ptr;             // 若在DRAM（pinned host memory）
    off_t ssd_offset;           // 若在SSD（Phoenix文件偏移）
    size_t ssd_length;          // SSD上的数据长度
};

// 扩展后的 vecflow index
template <typename data_t>
struct tiered_index {
    // 索引类型
    tagore::graph_index<data_t, uint32_t> ivf_graph_index;  // 高选择性（Tagore图索引）
    cuvs::neighbors::ivf_flat::index<data_t, int64_t> ivf_bfs_index;  // 低选择性（BFS）

    // 新增：分级存储管理
    std::vector<label_meta> label_metadata;   // 所有标签的元数据（常驻HBM）
    LabelCache hbm_cache;                     // HBM缓存池（标签级LRU）
    phxfs_fileid_t ssd_file;                  // Phoenix SSD文件句柄

    // 原有元数据
    int specificity_threshold;
    raft::device_vector<uint32_t, int64_t> cat_freq;
};
```

---

## 2. Task 1：Tagore 构建集成（预计 2-3 周）

> 目标：用 Tagore 全面替换 cuVS CAGRA，作为高选择性标签的图索引。构建用 Tagore 的 GNN-Descent + CFS 剪枝，搜索内核需适配 Tagore 图结构。

### 2.1 适配 Tagore 的内存接口

**当前问题**：Tagore 的 API 是基于文件的（`GNN_descent` 读 `.fvecs` 文件），VecFlow 的 per-label 数据在 GPU 内存中。

**修改内容**：
- 文件：`thirdparty/Tagore/src/Tagore_src.cu`
- 新增函数：`GNN_descent_from_device(half* d_data, unsigned num, unsigned dim, unsigned K, unsigned iterations, unsigned* d_graph)`
  - 输入：GPU 上的 half 精度向量数组
  - 输出：GPU 上的 kNN 图
  - 跳过文件读取和 `cudaMemcpy(H→D)`，直接在 GPU 内存上操作

**类型转换**：
- VecFlow 用 `float`，Tagore 用 `half`
- 在 per-label 构建前，用现有的 `f2h` kernel（`large_index.cu:117`）做一次转换
- 转换代价可忽略（一次性，相比构建本身是 O(1)）

### 2.2 模板化编译期常量

**当前问题**：Tagore 用 `#define` 固定了 `K_SIZE=96`, `DIM_SIZE=128` 等常量。

**修改内容**：
- 文件：`thirdparty/Tagore/include/Tagore_src.cuh`
- 将关键常量改为模板参数或运行时参数
- 最小改动方案：保留 `#define` 但通过 CMake 编译选项覆盖，为不同 (K, DIM) 组合编译多个版本

### 2.3 替换 VecFlow 构建路径

**修改内容**：
- 文件：`cpp/src/neighbors/vecflow/vecflow_build.cuh`
- 在 `#pragma omp parallel for` 循环内（约第 172-218 行），将：
  ```cpp
  auto index = cagra::build(thread_resources, index_params,
                            raft::make_const_mdspan(filtered_dataset.view()));
  ```
  替换为：
  ```cpp
  // 1. float → half 转换
  auto half_data = convert_to_half(filtered_dataset);
  // 2. GNN-Descent 构建 kNN 图
  auto knn_graph = tagore::GNN_descent_from_device(half_data, num_points, dim, K, iterations);
  // 3. CFS 剪枝（选择最优图结构，如 Vamana/NSG 等）
  auto pruned_graph = tagore::Pruning_from_device(knn_graph, ..., index_type, threshold);
  ```

### 2.4 适配 Tagore 图搜索内核

**核心问题**：原 VecFlow 用 cuVS 的 `cagra::filtered_search()` 做图遍历搜索。替换为 Tagore 图索引后，需要一个适配 Tagore 图结构的搜索内核。

**方案选择**：

| 方案 | 描述 | 优劣 |
|------|------|------|
| **(a) 复用 CAGRA 搜索内核** | Tagore 输出 CAGRA 兼容的邻接表格式 | 最快落地，但无法利用 Tagore 图结构优势 |
| **(b) 基于 Tagore 图写新搜索内核** | 针对 Tagore 的 NSG/Vamana 等图结构写专用 GPU 搜索 kernel | 性能最优，工作量大 |
| **(c) 分阶段** | 先 (a) 验证整体流程，再 (b) 优化搜索性能 | 推荐 |

**推荐方案 (c)**：
- 阶段一：Tagore 输出 CAGRA 兼容格式（`select_1hop_cagra`），搜索仍用 cuVS kernel，快速跑通
- 阶段二：为 Tagore 图结构（如 Vamana 的有向图、NSG 的导航图）实现专用 filtered_search kernel
  - 可以从 cuVS CAGRA 的 `filtered_search_single_cta` kernel 改写
  - 主要差异：entry point 选择（NSG/Vamana 有 medoid，CAGRA 无）、图遍历策略

**修改文件**：
- `cpp/src/neighbors/vecflow/vecflow_search.cuh`：替换 `cagra::filtered_search()` 调用
- 新增：`cpp/src/neighbors/vecflow/tagore_search_kernel.cuh`：Tagore 图的 GPU 搜索内核

### 2.5 多GPU并行构建

**修改内容**：
- 文件：`cpp/src/neighbors/vecflow/vecflow_build.cuh`
- 参考 Tagore 的 `gpu_construct()`（`large_index.cu:130`）的多GPU分配策略
- 将 OpenMP 并行循环改为多GPU感知：
  ```cpp
  #pragma omp parallel for num_threads(num_gpus * threads_per_gpu)
  for (uint32_t i = 0; i < label_number; i++) {
      int gpu_id = omp_get_thread_num() % num_gpus;
      cudaSetDevice(gpu_id);
      // ... Tagore 构建 ...
  }
  ```

### 2.6 验证

- 用相同数据集（YFCC-10M）对比：Tagore图索引 vs 原cuVS CAGRA
- 验证指标：
  - 构建时间对比（Tagore GNN-Descent vs cuVS NN-Descent）
  - 搜索 recall@10 对比（Tagore 图可能因不同剪枝策略而有不同 recall-QPS tradeoff）
  - 不同图类型的 recall-QPS pareto 曲线（NSG vs Vamana vs CAGRA格式）
  - 多GPU构建的加速比

**关键研究问题**：哪种 Tagore 图结构（NSG/Vamana/NSSG/DPG）最适合 per-label 的小子图 + 过滤搜索？这本身是一个论文贡献点。

---

## 3. Task 2：Phoenix 分级存储（预计 3-4 周）

> 目标：实现 SSD-DRAM-HBM 三级存储，支持超出 HBM 容量的数据集。

### 3.1 Phoenix 库集成

**修改内容**：
- 文件：`vecflow/CMakeLists.txt`
- 添加 Phoenix 库的编译和链接：
  ```cmake
  add_subdirectory(${CMAKE_SOURCE_DIR}/../thirdparty/phoenix/libphoenix phoenix_lib)
  target_link_libraries(vecflow PRIVATE phoenix)
  ```
- 需要确保 Phoenix 内核模块已加载（`sudo make insmod`）

### 3.2 标签级存储管理器

**新增文件**：`cpp/src/neighbors/vecflow/tiered_storage.cuh`

核心组件：

#### 3.2.1 LabelCache（HBM 缓存池）

```
功能：管理 HBM 中的标签数据缓存
策略：Size-aware LRU
      - 淘汰时优先淘汰 data_size 大且 access_count 低的标签
      - 公式：evict_priority = data_size / (access_count + 1)
接口：
      - load(label_id) → 从 DRAM/SSD 加载到 HBM
      - evict(label_id) → 从 HBM 淘汰到 DRAM/SSD
      - lookup(label_id) → 返回 HBM 指针或 nullptr
容量：可配置，默认占用 80% HBM
```

#### 3.2.2 SSD Manager（Phoenix I/O 封装）

```
功能：封装 Phoenix API，提供标签粒度的 SSD 读写
接口：
      - save_label(label_id, data, size) → 写入 SSD
      - load_label_async(label_id, dst_buf, stream) → 异步读到 HBM/DRAM
      - prefetch_labels(label_ids, stream) → 批量预取
底层：
      - phxfs_open() 初始化
      - phxfs_regmem() 注册 GPU 内存
      - phxfs_read_async() + CUDA 流做异步 DMA
```

### 3.3 构建时的分级存储

**修改内容**：`cpp/src/neighbors/vecflow/vecflow_build.cuh`

构建完成后，不是把所有标签数据都留在 HBM，而是：
1. 计算每个标签的 `data_size`
2. 按 `data_size` 降序排列标签
3. 将标签从大到小填入 HBM，直到 HBM 满
4. 剩余标签按大小分到 DRAM（中等）和 SSD（大且可能冷的）
5. 初始分配是静态的，运行时根据访问频率动态调整

### 3.4 搜索时的分级路由

**修改内容**：`cpp/src/neighbors/vecflow/vecflow_search.cuh`

在现有的 `classify_queries_kernel` 之后，新增一步 tier 路由：

```
原始流程：query → 按标签分类 → CAGRA/BFS搜索
新流程：  query → 按标签分类 → 按 tier 分组：
                                ├── HBM组：立即搜索（Stream 0）
                                ├── DRAM组：PCIe H2D 拷贝 → 搜索（Stream 1）
                                └── SSD组：Phoenix async DMA → 搜索（Stream 2）
```

关键优化：**三个 stream 流水线重叠**
- Stream 0 搜索 HBM 热标签（零等待）
- Stream 1 同时从 DRAM 预取温标签
- Stream 2 同时从 SSD 预取冷标签
- 各 stream 的搜索在数据就绪后依次启动

### 3.5 动态 tier 升降级

搜索过程中持续统计每个标签的 `access_count`。周期性（每 N 次查询批次后）触发：

```
for each label L:
    value(L) = access_count(L) / data_size(L)

排序后重新分配 tier：
    top K 个 value → HBM（升级）
    被挤出的 → DRAM/SSD（降级）
```

升降级通过 Phoenix async I/O 在后台完成，不阻塞当前搜索。

### 3.6 验证

- 数据集：YFCC-10M（约 3.8GB 向量 + 图索引）
- 模拟 HBM 容量受限（只允许 50%/25%/10% 数据在 HBM）
- 验证指标：
  - 搜索 QPS 对比（全HBM vs 分级存储）
  - Recall@10 一致性（应完全相同）
  - 冷标签首次访问延迟
  - Phoenix DMA 带宽利用率

---

## 4. Task 3：多GPU线性扩展（预计 2-3 周）

> 目标：N 张 GPU 实现接近 N 倍 QPS。
>
> **开发环境**：本机 2×A100 (80GB)。先在 A100 上验证多GPU架构和线性扩展性，后续迁移到消费级显卡（RTX 4090）做成本对比实验。

### 4.1 标签到GPU的分配策略

**新增文件**：`cpp/src/neighbors/vecflow/multi_gpu.cuh`

```
输入：所有标签的 data_size 和预估 access_frequency
输出：label → gpu_id 的映射

策略：加权负载均衡
      - 目标：每张 GPU 的 Σ(access_frequency × data_size) 尽量相等
      - 约束：每张 GPU 的 Σ(data_size) ≤ HBM_capacity × 80%　　// A100: 80GB × 80% = 64GB per GPU
      - 算法：贪心 bin-packing（按 access_frequency × data_size 降序放置）
```

### 4.2 查询路由器

**修改内容**：`cpp/src/neighbors/vecflow/vecflow_search.cuh`

```
输入：一批 queries + query_labels
流程：
      1. 按 label 查找目标 GPU（label_to_gpu 映射表）
      2. 按 GPU 分组 queries
      3. 每组通过 cudaMemcpyAsync(H2D) 发送到对应 GPU
      4. 各 GPU 并行搜索
      5. 收集结果，按原始 query 顺序合并
```

零跨GPU通信：单标签查询只涉及一张GPU。结果合并在 host 侧完成。

### 4.3 per-GPU 独立 Phoenix 实例

每张GPU独立初始化 Phoenix：
```cpp
for (int gpu_id = 0; gpu_id < num_gpus; gpu_id++) {
    cudaSetDevice(gpu_id);
    phxfs_open(gpu_id);
    // 注册该 GPU 的 HBM 缓存区
    phxfs_regmem(gpu_id, hbm_cache_ptr, cache_size, &target);
}
```

各GPU独立访问SSD，不竞争。

### 4.4 验证

**开发阶段**（本机 2×A100）：
- 配置：1 × A100 vs 2 × A100
- 数据集：YFCC-10M, YFCC-100M（如可用）
- 验证指标：
  - QPS: 2×A100 是否接近 2× 单A100 QPS
  - 单标签查询 vs 全局查询的扩展性对比
  - 负载均衡效果（两张GPU利用率对比）
  - 架构正确性：查询路由、结果合并、分级存储跨GPU一致性

**论文实验阶段**（后续迁移）：
- 配置：1/2/4/8 张 RTX 4090
- 核心实验：
  - QPS vs GPU 数量曲线（期望接近线性）
  - 成本对比：4×RTX 4090 (~$8K) vs 1×A100 (~$15K)
  - 消费级显卡 + Phoenix 分级存储的效果验证

---

## 5. Task 4：端到端集成与优化（预计 2 周）

### 5.1 统一 Python API

**修改内容**：`vecflow/include/vecflow.hpp`, `vecflow/src/vecflow.cu`

```python
from vecflow import VecFlow

vf = VecFlow(
    num_gpus=2,                    # 新增：GPU数量（开发环境 2×A100）
    hbm_budget_ratio=0.8,          # 新增：HBM缓存占比
    ssd_path="/data/vecflow.phx",  # 新增：SSD存储路径
    builder="tagore",              # 新增：构建器选择 ("tagore" | "cagra")
)

# 构建（自动多GPU + Tagore + 分级存储）
vf.build(dataset, data_labels,
         graph_degree=16,
         specificity_threshold=2000)

# 搜索（自动路由 + 分级加载）
neighbors, distances = vf.search(queries, query_labels, itopk_size=32)

# 新增：查看存储统计
vf.storage_stats()
# → HBM: 1200 labels (3.2GB), DRAM: 500 labels (1.8GB), SSD: 8300 labels (12.6GB)
```

### 5.2 CMake 构建系统更新

**修改内容**：`vecflow/CMakeLists.txt`

```cmake
# 新增依赖
option(VECFLOW_USE_PHOENIX "Enable Phoenix tiered storage" ON)
option(VECFLOW_USE_TAGORE "Enable Tagore graph builder" ON)

if(VECFLOW_USE_PHOENIX)
    add_subdirectory(${CMAKE_SOURCE_DIR}/../thirdparty/phoenix/libphoenix phoenix_lib)
    target_link_libraries(vecflow PRIVATE phoenix)
    target_compile_definitions(vecflow PRIVATE VECFLOW_PHOENIX_ENABLED)
endif()

if(VECFLOW_USE_TAGORE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/../thirdparty/Tagore tagore_lib)
    target_link_libraries(vecflow PRIVATE tagore)
    target_compile_definitions(vecflow PRIVATE VECFLOW_TAGORE_ENABLED)
endif()
```

### 5.3 性能调优

- **预取深度**：根据 SSD 带宽和查询批次大小调整预取标签数
- **HBM 缓存大小**：根据标签大小分布的长尾特性自动调整
- **Tagore 迭代次数**：根据 per-label 数据量自适应选择 GNN-Descent 迭代轮数
- **流水线深度**：调整 Stream 0/1/2 的并发度

---

## 6. Task 5：实验设计与论文（预计 3-4 周）

### 6.1 实验对比

| 实验 | Baseline | Proposed | 指标 |
|------|----------|----------|------|
| E1: 单GPU QPS | VecFlow(CAGRA) on A100 | VecFlow(Tagore)+Phoenix on A100 | QPS, Recall@10 |
| E2a: 多GPU扩展性(开发) | Proposed on 1×A100 | Proposed on 2×A100 | QPS 2×接近线性验证 |
| E2b: 多GPU扩展性(论文) | Proposed on 1×RTX 4090 | Proposed on 1/2/4/8×RTX 4090 | QPS vs #GPU 曲线 |
| E3: 超HBM数据集 | VecFlow OOM | Proposed 分级存储 | 最大可处理数据集大小 |
| E4: 构建速度 | cuVS CAGRA build | Tagore build (GNN-Descent + CFS) | 构建时间, 多GPU加速比 |
| E4b: 图结构对比 | CAGRA格式 | NSG/Vamana/NSSG/DPG | 不同图结构的filtered recall-QPS pareto |
| E5: 冷热自适应 | 静态分配 | 动态tier调整 | Zipf分布下的QPS变化 |
| E6: 成本效益 | 1×A100 (~$15K) | 4×RTX 4090 (~$8K) | QPS/$ |
| E7: A100 vs RTX 4090 | 2×A100 | N×RTX 4090 (HBM等效) | 同等HBM容量下的QPS对比 |

### 6.2 数据集

| 数据集 | 规模 | 维度 | 标签数 | 用途 |
|--------|------|------|--------|------|
| YFCC-10M | 10M | 192 | ~10K | 主实验 |
| YFCC-100M | 100M | 192 | ~10K | 扩展性 |
| SIFT-1B | 1B | 128 | 合成标签 | 极限压测 |

### 6.3 论文结构（建议）

```
Title: "Scaling Filtered ANNS to Consumer GPUs with Tiered Storage and GPU-Optimized Graph Construction"

1. Introduction
   - 动机：数据中心GPU太贵，消费级GPU的HBM不够
   - 贡献：三个正交技术的整合

2. Background & Motivation
   - VecFlow: 选择性与双索引
   - 关键观察：选择性 ⊥ 访问频率（正交性）

3. System Design
   3.1 双维分类策略（选择性 × 访问频率）
   3.2 Phoenix-based 三级存储
   3.3 Tagore-based 多GPU图构建
   3.4 标签级查询路由

4. Implementation
   4.1 LabelCache: Size-aware LRU
   4.2 Phoenix 异步 DMA 流水线
   4.3 多 GPU 负载均衡

5. Evaluation
   5.1 单GPU QPS（E1）
   5.2 线性扩展性（E2）
   5.3 超HBM容量（E3）
   5.4 构建速度（E4）
   5.5 动态自适应（E5）
   5.6 成本效益（E6）

6. Related Work
7. Conclusion
```

---

## 7. 里程碑时间线

| 阶段 | 时间 | 交付物 |
|------|------|--------|
| **Task 1**: Tagore 图索引集成 | 第 1-4 周 | Tagore per-label 构建可用，搜索内核适配完成，多GPU构建可用，recall 验证通过 |
| **Task 2**: Phoenix 分级存储 | 第 4-7 周 | SSD-DRAM-HBM 三级可用，动态 tier 升降级可用 |
| **Task 3**: 多GPU扩展 | 第 8-10 周 | 标签路由、per-GPU Phoenix、负载均衡可用 |
| **Task 4**: 端到端集成 | 第 11-12 周 | 统一 Python API、CMake 构建、性能调优 |
| **Task 5**: 实验与论文 | 第 13-16 周 | 完整实验数据、论文初稿 |

---

## 8. 风险与缓解

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| Tagore 图搜索内核开发 | 高 | 阶段一复用 CAGRA 兼容格式 + cuVS 搜索内核快速跑通；阶段二开发专用 kernel |
| Phoenix 内核模块在某些 kernel 版本上不兼容 | 中 | 固定 Ubuntu 22.04 + Linux 6.1，已验证环境 |
| Tagore 编译期常量限制灵活性 | 中 | 先 CMake 多版本编译，后续模板化 |
| 多GPU负载不均衡 | 中 | Zipf 分布下，热标签少 → work stealing 兜底 |
| 冷标签首次访问延迟过高 | 中 | 查询攒批 + 预取窗口，与热标签搜索重叠 |
| 分级存储下图遍历随机访问延迟 | 高 | 批量预取整个标签子图到 HBM 缓存后再搜索，不做 fine-grained page fault |

---

## 9. 关键文件索引

| 组件 | 关键文件 |
|------|----------|
| VecFlow 构建 | `cpp/src/neighbors/vecflow/vecflow_build.cuh` |
| VecFlow 搜索 | `cpp/src/neighbors/vecflow/vecflow_search.cuh` |
| Tagore 搜索内核（新增） | `cpp/src/neighbors/vecflow/tagore_search_kernel.cuh` |
| VecFlow 公共 | `cpp/src/neighbors/vecflow/vecflow_common.cuh` |
| VecFlow 索引定义 | `cpp/include/cuvs/neighbors/vecflow.hpp` |
| VecFlow Python 绑定 | `vecflow/src/vecflow.cu`, `vecflow/include/vecflow.hpp` |
| Phoenix 用户库 | `thirdparty/phoenix/libphoenix/phoenix.cc` |
| Phoenix API 头文件 | `thirdparty/phoenix/libphoenix/include/phoenix.h` |
| Phoenix CUDA 集成 | `thirdparty/phoenix/libphoenix/integration.cc` |
| Phoenix 内核模块 | `thirdparty/phoenix/module/phxfs.c` |
| Tagore GPU 内核 | `thirdparty/Tagore/src/Tagore_src.cu` |
| Tagore 大规模索引 | `thirdparty/Tagore/src/large_index.cu` |
| Tagore 公共函数 | `thirdparty/Tagore/src/common.cu` |
| CMake 构建 | `vecflow/CMakeLists.txt` |
