# 最终冻结审查记录

生成时间：2026-05-31

## 1. 已完成项

| 冻结条件 | 证据 |
|---|---|
| 严格递推主指标没有稳定新提升 | `outputs/q4_candidate_model_review.md` 显示近 28 日均值候选全样本 WAPE=74.95%，但留一窗口改善仅 0.19 个百分点，未替换当前混合策略 |
| 主结论均有表格或报告来源 | `tables/q4_low_volume_strategy_metrics.csv`、`tables/q4_candidate_model_metrics.csv`、`tables/final_forecast_interval_by_store.csv` 等 |
| PDF 可编译 | `paper/final_paper.pdf`、`submission/final_paper.pdf` 均为 102 页 A4；最新日志 `paper/latex/compile_latexmk_20260531.log` |
| 摘要、正文、结论预测表口径一致 | 论文使用 `outputs/final_7day_forecast_hybrid_low_volume.csv` 作为候选最终预测表，总量 1329.894；提交包同口径 |
| 无因果化主结论 | 正文使用“统计关联”“条件预测”“情景假设”等表述 |
| 无夸大主结论 | 近 28 日均值候选降调为探索性候选；综合 Ridge 未写成严格递推显著优于强 baseline |
| 答辩材料覆盖关键问题 | `outputs/defense_notes.md` 已覆盖复杂模型、Ridge 保留、混合策略、未来天气/活动日、WAPE 高的解释 |
| 提交包已同步混合策略口径 | `submission/final_7day_forecast.csv` 与 `outputs/final_7day_forecast_hybrid_low_volume.csv` 哈希一致；整数表同理 |
| 封面成员真实信息已填写 | `paper/latex/main.tex` 已写入 25011076 吴渔桐、25012721 张哲凌、25010409 朱饶杰三位成员信息；`paper/final_paper.pdf` 与 `submission/final_paper.pdf` 已重新编译同步 |

## 2. 仍需赛前人工确认项

| 项目 | 当前状态 | 原因 |
|---|---|---|
| 封面成员真实信息 | 已按用户提供信息填写 | 正式上传前仍应与报名系统核对姓名、学号、学院、班级和电话是否一致 |
| 是否匿名提交 | 当前版本为非匿名封面 | 若竞赛平台临时要求匿名，应运行 `C:\ProgramData\anaconda3\python.exe src\fill_cover_members.py --anonymous` 后重新编译同步 PDF |
| 平台提交格式 | 材料已同时提供小数表和整数表 | 需按平台要求选择小数表或整数表、每日明细或 7 日汇总 |

## 3. 当前建议

模型优化、论文口径、图表一致性、预测区间、提交包同步、封面成员信息和 PDF 编译检查均已完成。正式上传前仅需团队按竞赛平台规则选择预测表口径，并核对封面信息与报名系统一致。
