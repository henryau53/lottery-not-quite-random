# Feature 第一阶段实现

本目录按 `PROJECT_CONTEXT.md` 与 `FEATURE_DESIGN.md` 的第一阶段设计实现。

## 输出

```text
data/dlt/features/
├── draw_features.parquet
├── number_features.parquet
├── prize_features.parquet
└── feature_metadata.json
```

## 运行

在项目根目录：

```bash
python -m lottery_not_quite_random.features
```

或：

```bash
python -m lottery_not_quite_random.features \
  --processed-dir data/dlt/processed \
  --output-dir data/dlt/features
```

## 测试

```bash
pytest tests/test_features.py
```

## 设计边界

- 只读取 `data/dlt/processed/`。
- 不修改 processed。
- 时间方向统一为旧 → 新。
- 预测型 Feature 使用当前期之前的信息。
- Rolling 统一先 `shift(1)`，再做窗口统计。
- 第一阶段不生成 P2 Distribution / co-occurrence 等扩展 Feature。
- Prize-level 第一阶段保持 prize record 粒度，重点完成 Schema、输出和 Context 关联。
- `prize_event_type` 不被错误聚合成 Draw-level 单值。

## 注意

有一个我刻意没有“自行补设计”的地方：std 的统计定义在文档里只规定了 std，没有明确 ddof=0 还是 ddof=1。当前代码采用 pandas 默认的 ddof=1，但这属于实现层选择，正式进入 Analysis 前建议把这个定义写进 FEATURE_DESIGN.md，否则未来不同实现可能产生数值差异。这符合项目“定义、时间语义、Metadata 和代码必须一致”的原则。
