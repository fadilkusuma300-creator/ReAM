# ReAM：面向不完整与冲突地理证据的可靠性感知地址匹配

本仓库提供 ReAM（Reliability-Aware Address Matching）的实现。模型显式区分证据“是否可用”与“当前候选对上是否可靠”，分别编码文本、空间和邻域证据，并使用候选对级的分支正确性概率作为融合权重依据。

## 参考实验环境

本仓库使用如下固定运行环境：

| 项目 | 固定配置 |
|---|---|
| 操作系统 | Ubuntu 22.04 LTS 64-bit |
| Python | 3.10.14 |
| PyTorch | 2.3.1 |
| CUDA | 12.1 |
| Transformers | 4.41.2 |
| NumPy | 1.26.4 |
| scikit-learn | 1.5.0 |
| PyYAML | 6.0.1 |
| tqdm | 4.66.4 |
| GPU | 1 × NVIDIA RTX A6000 48 GB |
| 混合精度 | BF16 |

推荐直接使用：

```bash
conda env create -f environment.yml
conda activate ream
```

也可以使用固定依赖文件：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

## 方法实现

实现配置如下：

- 文本、空间、邻域三个证据分支；
- `bert-base-uncased`，最大长度 128，三个分支表示维度均为 128；
- 邻域半径 500 m，每侧最多保留 15 个邻居；
- 邻居名称字符 CNN：卷积核 2/3/4/5，每种 32 通道，字符嵌入 32 维；
- 相对距离使用 16 个 RBF；
- 双向软邻域对应，默认 `beta=0.2`、`T=1.0`；
- 两层可靠性头，隐藏维度 64，GELU，dropout 0.1；
- availability-aware reliability normalization；
- 前 5 个 epoch 只训练三个分支，之后进入联合可靠性训练；
- 每个 epoch 刷新分支 correctness target；
- 总损失 `L_match + 0.3 L_branch + 0.2 L_rel`；
- AdamW，文本编码器学习率 `2e-5`，其他模块 `1e-4`，weight decay 0.01；
- batch size 64，最多 25 epoch，patience 5；
- ECE、Brier、risk-coverage、AURC 与分支可靠性校准；
- full ReAM、static average、Attention+CD、Confidence+CD、w/o reliability supervision、w/o quality features、w/o controlled degradation 等变体。

## Geo-ER 数据

实验使用公开 Geo-ER benchmark，不在本仓库重复分发原始数据。使用官方固定的 train/validation/test 划分，候选对不做重新划分。

八个任务为：

- Singapore：OSM-Foursquare、OSM-Yelp；
- Edinburgh：OSM-Foursquare、OSM-Yelp；
- Toronto：OSM-Foursquare、OSM-Yelp；
- Pittsburgh：OSM-Foursquare、OSM-Yelp。

数据目录：

```text
data/prepared/
  osm_fsq/{singapore,edinburgh,toronto,pittsburgh}/{train,valid,test}.jsonl
  osm_yelp/{singapore,edinburgh,toronto,pittsburgh}/{train,valid,test}.jsonl
```

转换示例：

```bash
python scripts/prepare_geoer.py pairs \
  --input data/raw/train.txt \
  --output data/prepared/train.jsonl \
  --city singapore \
  --source-pair osm_fsq
```

## 训练协议

每个主任务执行 5 次独立训练。固定数据划分不变，仅改变模型初始化、mini-batch 顺序与退化随机状态。种子：`13, 29, 47, 71, 101`。

```bash
python scripts/train.py \
  --train data/prepared/train.jsonl \
  --valid data/prepared/valid.jsonl \
  --output outputs/sin_osm_fsq
```

训练设置：

- 最大 25 epoch；
- 前 5 epoch 为 branch-only warm-up；
- validation F1 早停，patience=5；
- batch size=64；
- BERT 学习率 `2e-5`；
- 其他模块学习率 `1e-4`；
- AdamW weight decay=0.01；
- dropout=0.1；
- branch correctness threshold=0.5；
- 最终 matching threshold 只在 validation set 上选择。

## 受控证据退化

训练采样总体比例为：50% 原始证据、35% 单证据退化、15% 双证据复合退化。

单证据退化范围：

- 文本字段遮蔽：0.10–0.40；
- 坐标偏移：20–200 m；
- 邻域删除：0.20–0.60；
- 属性冲突：从同一 city、source pair、split 的非匹配 donor 中替换，排除真实匹配实体。

复合条件为：30% 文本遮蔽 + 100 m 坐标偏移。

```bash
python scripts/degradation_eval.py --data data/prepared/test.jsonl --kind text_mask --output-dir data/degraded/text
python scripts/degradation_eval.py --data data/prepared/test.jsonl --kind coordinate_shift --output-dir data/degraded/spatial
python scripts/degradation_eval.py --data data/prepared/test.jsonl --kind neighborhood_delete --output-dir data/degraded/neighborhood
python scripts/degradation_eval.py --data data/prepared/test.jsonl --kind attribute_conflict --output-dir data/degraded/conflict
```

## 测试与指标

```bash
python scripts/evaluate.py \
  --data data/prepared/test.jsonl \
  --checkpoint outputs/sin_osm_fsq/best.pt \
  --output outputs/sin_osm_fsq/test_metrics.json
```

主要输出包括 F1、15-bin ECE、Brier、AURC、风险-覆盖率数据、branch probability、predicted reliability、normalized fusion weight 与 branch calibration bins。

## 消融与跨城市

```bash
bash scripts/run_ablation.sh data/prepared/train.jsonl data/prepared/valid.jsonl outputs/ablation
bash scripts/run_cross_city.sh data/prepared outputs/cross_city
```

LOCO 实验中，被留出的城市只使用官方 test partition；训练统计量、归一化参数、典型空间尺度、超参数和最终阈值均由其余三个训练城市的数据确定。

## 关键可复核约束

- Geo-ER 官方候选集及固定 split 保持不变；
- 不做正负样本重采样，也不使用 class weighting；
- spatial scale 与标准化统计量只由相应 training partition 拟合；
- neighborhood deletion 只改变邻域分支的当前邻居集合；空间分支的 density 取自 `spatial_neighbor_count`，不随邻居删除变化；
- robustness evaluation 只改变目标证据状态，不修改 candidate set 和 ground-truth label；
- 正常 inference 不注入 degradation；
- attribute-conflict donor 限制在当前数据 partition 内，并排除真实匹配实体。

## 测试

```bash
pytest -q
```

## 目录结构

```text
configs/                 配置文件
scripts/                 数据准备、训练、测试、鲁棒性、消融、跨城市入口
src/ream/                ReAM 核心实现
tests/                   单元测试
requirements.txt         最低依赖范围
requirements-lock.txt    固定发布依赖
environment.yml          参考 Conda 环境
README.md                 英文主页
README_CN.md              中文说明
```

## 引用

如使用 Geo-ER benchmark，请引用 Geo-ER 原始论文及其公开仓库。
