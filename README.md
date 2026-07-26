# LCC GRU-MPC

本项目使用四维横向自行车模型描述主要车辆动力学，使用GRU预测模型残差，再将
扰动预测交给MPC做前馈补偿。GRU不是独立方向盘控制器。

## 当前推荐配置

- 状态：`[e_y, e_y_dot, e_psi, e_psi_dot]`
- GRU输入：`[d_e1dot, e1, e1dot, e2, e2dot, front_steer, vx]`
- 历史窗口：20步（1秒）
- 学习预测：10步（0.5秒）
- MPC扰动序列：25步；后15步暂时保持第10步预测值
- 状态差分：因果后向差分 + 一阶低通

`kappa`和`steer_rate`的原始直接输入在现有数据上存在严重分布外放大，当前不属于
推荐配置；相关结论保留在实验报告中。

## 标准流程

```bash
conda run --no-capture-output -n ml python data/classify_highway_data.py
conda run --no-capture-output -n ml python data/dataset.py
conda run --no-capture-output -n ml python model/train_gru.py --device cpu
```

默认产物：

- `processed/highway_data/`：可审计分类manifest和集合清单
- `processed/highway_dataset.npz`：当前推荐7维、10步因果数据集
- `processed/highway_gru/`：最佳checkpoint、开发指标和损失曲线

`stress_final` 是冻结终测集，不应在日常调参中重复查看。方案冻结后才运行
`model/evaluate_gru.py`，将最终指标写入 `processed/highway_gru/final_stress_metrics.json`。

## 当前结论

正常测试总体相对扰动保持基线改善约4.24%，但20–25和25–30 m/s分别恶化约
6.6%和1.6%。工程链路已经因果化并具备MPC接口，正常高速收益仍未达到上车条件。

详细过程见 [2026-07-11实验报告](docs/2026-07-11-highway-gru-report.md)。
