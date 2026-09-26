# 正式训练与结果汇总已完成

正式运行：`results/20260923-153024-429172-train/`。

BCE/BPR 各三个种子、30 epoch，共六组训练；完整数据 6,040 位用户，固定评估 3,518 位用户。全部训练与后续复算通过。

首先阅读该运行下 `analysis/RESULTS.md` 和 `analysis/INTERPRETATION.md`。正式结果、CSV、PNG/PDF 和 LaTeX 表格已生成；原始分数及逐用户诊断保存在 `train/`。结果汇总可通过以下命令重建：

```powershell
.\.venv\Scripts\python.exe analyze_results.py results/20260923-153024-429172-train
```

以下保留先前的理论核验与端到端测试记录。

# 已完成的运行

本文件记录项目交付时的实际完成状态，不将测试配置当作正式训练结果。

最终版本已通过 8 项正确性测试。最终端到端验证位于 `results/20260923-152745-318931-all/`，额外包含可复现的 `source_snapshot.zip` 与依赖版本记录；结果与此前成功运行一致。200 位抽样用户中，111 位满足共同验证/测试评估条件。

* `results/20260923-152330-738866-core/`：正式理论配置。枚举 32,738 个二元排列，22,750 项同 cutoff / 全局常数检查与 6,930 项跨 cutoff 检查通过；闭式常数与枚举最大误差约 7.8e-12。60 组尺度设置通过达到边界的排列核验，受控优化实验完成。
* `results/20260923-152525-693671-all/`：全部模块的端到端测试。MovieLens 1M 中抽取 200 位用户，BCE/BPR 各运行一个种子、3 个 epoch；包括 epoch 0 初始状态。完整分数、切分、轨迹、置信区间及图已归档。
* `results/20260923-152005-702798-all/`：首次尝试因 Python 证书链导致官方数据下载失败，保留失败日志。随后通过 Windows 系统证书验证下载完成，未禁用 HTTPS 验证。

正式的全用户、三种子、30 epoch 训练现已完成，见下文的正式结果目录；smoke 运行仍只作为实现核验。

建议先查看成功运行下的 `train/position_and_conflict.png`、`train/training_trajectories.png`，以及正式理论运行下的 `scaling/scaling.png`。原始结果和对应运行时源码哈希保存在各自目录，不以之后代码修改替换历史记录。
