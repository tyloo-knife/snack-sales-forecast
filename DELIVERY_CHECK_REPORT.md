# DELIVERY_CHECK_REPORT

检查日期：2026-06-01

## 一、执行原则

本轮为交付前确认与打包。未修改任何建模脚本、结果数字、论文正文措辞或参赛人员信息页；仅新增交付检查报告和电子附件打包产物。

## 二、Phase 0：全流程零崩溃复现

复现环境：

- 虚拟环境：`C:\Users\20215\AppData\Local\Temp\snack_sales_phase0_venv_20260601`
- Python：3.13.9
- `pip check`：通过，`No broken requirements found`
- `requirements.txt`：未改动，未补包
- 复现工作副本：`C:\Users\20215\AppData\Local\Temp\snack_sales_phase0_work_20260601`

脚本退出码：

| 脚本 | 退出码 | Traceback |
|---|---:|---|
| `src/stage2_q1_figures.py` | 0 | 无 |
| `src/stage6_phase3_structure_analysis.py` | 0 | 无 |
| `src/stage7_phase4_text_evidence.py` | 0 | 无 |
| `src/stage3_q2_analysis.py` | 0 | 无 |
| `src/stage4_q3_analysis.py` | 0 | 无 |
| `src/stage5_q4_analysis.py` | 0 | 无 |
| `src/stage5_q4_recursive_validation.py` | 0 | 无 |
| `src/stage5_q4_ablation.py` | 0 | 无 |
| `src/stage5_q4_low_volume_strategy.py` | 0 | 无 |
| `src/stage5_q4_hybrid_final_forecast.py` | 0 | 无 |
| `src/stage5_q4_error_diagnosis.py` | 0 | 无 |
| `src/stage5_q4_weather_sensitivity.py` | 0 | 无 |
| `src/intermittent_demand_review.py` | 0 | 无 |

重点输出确认：

| 文件 | 结果 |
|---|---|
| `outputs/stage3_q2_summary.json` | 成功写出 |
| `outputs/stage4_q3_summary.json` | 成功写出 |
| `outputs/stage5_q4_summary.json` | 成功写出 |

## 三、Phase 1：论文编译与终值对账

论文编译：

| 项目 | 结果 |
|---|---|
| 编译命令 | `latexmk -xelatex -interaction=nonstopmode main.tex` |
| 编译目录 | `paper/latex/` |
| 退出码 | 0 |
| PDF | `paper/latex/main.pdf` |
| 页数 | 44 |
| 编译错误 | 无 |
| undefined reference/citation | 无 |
| 缺图 | 无 |
| Overfull | 无 |
| Underfull | 有若干提示，不构成编译错误或缺图问题 |

终值对账：

| 对账项 | 对账结果 | 判定 |
|---|---:|---|
| 全文/PDF 搜索 `1328/1330/1331` | 仅出现 `1328`，无 `1330/1331` | 通过 |
| 最终明细整数表 | 532 行，总计 1328 | 通过 |
| 门店汇总表 | 7 行，总计 1328 | 通过 |
| 类别汇总表 | 8 行，总计 1328 | 通过 |
| 商品汇总表 | 12 行，总计 1328 | 通过 |
| 门店--商品汇总表 | 76 行，总计 1328 | 通过 |
| 附录门店表 | 总计 1328 | 通过 |
| 附录门店--日期表 | 总计 1328 | 通过 |
| 附录类别表 | 总计 1328 | 通过 |
| 附录类别--日期表 | 总计 1328 | 通过 |
| 附录商品表 | 总计 1328 | 通过 |
| 附录门店--商品表 | 总计 1328 | 通过 |

表 11 对账：

| 项目 | 结果 | 判定 |
|---|---:|---|
| 编译编号 | 表 11 | 通过 |
| TeX 表行数 | 10 | 通过 |
| `tables/q3_regression_coefficients_external.csv` 行数 | 10 | 通过 |
| TeX 表变量集合与 CSV | 一致 | 通过 |
| 周末/is_weekend 行 | TeX 表与 CSV 均无 | 通过 |
| 样本量与调整 R² | `N=58517`，调整 `R^2=0.370` | 通过 |
| 活动日系数 | `0.325` | 通过 |
| 周末补充模型 | `0.065`，仅补充模型 | 通过 |

表 16 对账：

| 比较对象 | 论文值 | CSV 值 | 判定 |
|---|---|---|---|
| 综合 Ridge vs 问题一简单指数平滑 | `76.35 / 76.68 / -0.34 / p=0.461 / [-1.56, 1.51]` | 一致 | 通过 |
| 低销量混合策略 vs 问题一简单指数平滑 | `76.35 / 75.85 / 0.49 / p=0.0503 / [-0.72, 1.34]` | 一致 | 通过 |

附录 E 路径检查：

| 项目 | 结果 |
|---|---|
| `\path{...}` 条目 | 23 处 |
| 唯一路径 | 22 个 |
| 路径存在性 | 全部存在 |
| `outputs/*.md` 通配 | 匹配 17 个内部过程记录 |

图文件检查：

| 项目 | 结果 |
|---|---|
| `\includegraphics` 条目 | 16 个 |
| 图文件存在性 | 全部存在 |

## 四、Phase 2：电子附件打包

过程记录处置：

| 项目 | 结果 |
|---|---|
| `outputs/*.md` AI 痕迹关键词检索 | 未命中 |
| 电子附件是否纳入 `outputs/*.md` | 未纳入 |
| 与附录 E/README 声明 | 一致：`outputs/*.md` 仅作过程追溯，电子附件优先提交最终预测、关键表图和代码 |

个人信息终检：

| 项目 | 结果 |
|---|---|
| 论文源文件个人信息搜索 | 仅在 `paper/latex/main.tex` 指定封面成员信息表中出现 |
| PDF 第 2 页以后个人信息搜索 | 未命中 |
| 参赛人员信息页 | 未改动 |

电子附件目录：

`C:\Users\20215\Documents\CodexProjects\02-modeling\snack-sales-forecast\submission\delivery_package`

电子附件压缩包：

`C:\Users\20215\Documents\CodexProjects\02-modeling\snack-sales-forecast\submission\snack_sales_forecast_delivery_20260601.zip`

压缩包 SHA256：

`1ef559c9681c4f236b6233e495fc13766e5cf1435e46fda3943637b7dfb0aeaa`

附件清单：

| 文件 | 路径 |
|---|---|
| 包内清单 | `submission/delivery_package/ELECTRONIC_ATTACHMENT_MANIFEST.csv` |
| 提交目录副本 | `submission/ELECTRONIC_ATTACHMENT_MANIFEST.csv` |

包内文件统计：

| 类别 | 数量 |
|---|---:|
| 编译论文 PDF | 1 |
| 文档说明 | 3 |
| 最终整数预测明细 | 1 |
| 最终连续预测明细 | 1 |
| 关键结果图 | 36 |
| 关键结果表 | 130 |
| Python 源码 | 21 |
| 附件清单文件 | 1 |
| 合计 | 194 |

关键附件存在性：

| 文件 | 结果 |
|---|---|
| `final_paper.pdf` | 已纳入 |
| `README.md` | 已纳入 |
| `DATA_DICTIONARY.md` | 已纳入 |
| `requirements.txt` | 已纳入 |
| `outputs/final_7day_forecast_hybrid_low_volume_integer.csv` | 已纳入 |
| `outputs/final_7day_forecast_hybrid_low_volume.csv` | 已纳入 |
| `src/` 全部代码 | 已纳入 |
| `tables/q3_regression_coefficients_external.csv` | 已纳入 |
| `tables/q4_significance_tests.csv` | 已纳入 |
| `figures/q4_hybrid_forecast_7day_total_by_store.png` | 已纳入 |

包体核验：

| 项目 | 结果 |
|---|---|
| zip 可读 | 是 |
| zip entry 数 | 194 |
| zip 内 manifest | 存在 |
| zip 内 PDF | 存在 |
| zip 内 `outputs/*.md` | 不存在 |
| PDF 与 `paper/latex/main.pdf` 哈希 | 一致 |
| 最终整数预测明细与源文件哈希 | 一致 |
| 最终连续预测明细与源文件哈希 | 一致 |

## 五、最终结论

Phase 0、Phase 1、Phase 2 均通过。未发生真实脚本崩溃或论文编译失败，因此未进行任何代码修复、数字修复或论文措辞修复。交付包已生成，可使用 `submission/snack_sales_forecast_delivery_20260601.zip` 作为本轮电子附件。
