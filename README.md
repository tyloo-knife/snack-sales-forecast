# 休闲零食连锁店商品销量预测

本仓库为数学建模 A 题《休闲零食连锁店商品销量预测》的数据处理、建模分析和论文支撑材料。项目使用历史零售明细、天气、节假日和活动日数据，完成门店、商品、类别和门店-商品粒度的销量分析与未来 7 天预测。

## 目录结构

| 路径 | 内容 |
|---|---|
| `data/raw/` | 题目原文和原始附件，保持只读 |
| `data/processed/` | 清洗后的日销量面板和建模基础表 |
| `src/` | 数据读取、预处理、特征、模型评价和各问题分析脚本 |
| `tables/` | 核心统计表、验证结果和预测汇总 |
| `figures/` | 论文使用的主要图表 |
| `outputs/` | 阶段报告、最终预测表和专项分析报告 |
| `paper/latex/` | 论文 LaTeX 源文件 |
| `submission/` | 面向提交或审查的论文 PDF 与预测表 |

## 核心数据

原始附件位于 `data/raw/`，包括：

- `A-休闲零食连锁店商品销量预测.docx`
- `附件一：历史零售明细统计.xlsx`
- `附件二：某市近两年天气等数据.xlsx`

处理后的主要数据表：

- `data/processed/daily_store_product_sales.csv`
- `data/processed/modeling_base_table.csv`

字段含义和数据处理口径见 `DATA_DICTIONARY.md`。原始数据不在脚本中覆盖或改写。

## 建模方法

本项目按题目四个问题组织：

1. 问题一：使用移动平均、同星期均值和简单指数平滑分析并预测门店、商品和门店-商品销量。
2. 问题二：计算商品销量相关性，并按附件中的商品类别进行同类零食聚合预测。
3. 问题三：使用描述性统计、固定效应回归和随机森林置换重要性分析天气、节假日、周末和活动日与销量的统计关联。
4. 问题四：在门店-商品粒度上融合历史销量、日历、类别和外部变量，使用 Ridge、随机森林和基准模型进行严格 7 日递推验证。

主要评价指标为 MAE、RMSE 和 WAPE。时间序列验证不随机打乱，问题四以严格 7 日递推验证作为主口径。

## 主要结果文件

| 文件 | 说明 |
|---|---|
| `outputs/stage1_data_inspection_report.md` | 数据结构、缺失、重复和异常检查 |
| `outputs/stage2_q1_modeling_report.md` | 问题一销量分析与基础预测 |
| `outputs/stage3_q2_relationship_report.md` | 问题二商品关联与类别预测 |
| `outputs/stage4_q3_factor_analysis_report.md` | 问题三外部因素统计关联分析 |
| `outputs/stage5_q4_final_model_report.md` | 问题四综合模型报告 |
| `outputs/q4_recursive_7day_model_comparison.md` | 严格 7 日递推验证说明 |
| `outputs/q4_error_diagnosis_report.md` | 问题四误差诊断 |
| `outputs/q4_weather_sensitivity_report.md` | 天气情景敏感性检验 |
| `outputs/final_7day_forecast.csv` | 最终连续预测表 |
| `outputs/final_7day_forecast_integer.csv` | 最终整数预测表 |
| `paper/latex/main.tex` | 论文主文件 |
| `submission/final_paper.pdf` | 最终论文 PDF |
| `submission/final_7day_forecast.csv` | 提交用连续预测表 |
| `submission/final_7day_forecast_integer.csv` | 提交用整数预测表 |

## 复现方式

GitHub 仓库地址：[https://github.com/tyloo-knife/snack-sales-forecast](https://github.com/tyloo-knife/snack-sales-forecast)。

建议使用 Python 3.10 及以上版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

按下列顺序运行主要脚本：

```powershell
python src/stage3_q2_analysis.py
python src/stage4_q3_analysis.py
python src/stage5_q4_analysis.py
python src/stage5_q4_recursive_validation.py
python src/stage5_q4_error_diagnosis.py
python src/stage5_q4_weather_sensitivity.py
python src/intermittent_demand_review.py
```

脚本输出会写入 `outputs/`、`tables/` 和 `figures/`。问题一的结果已保存在对应报告、表格和图表中。

## 论文编译

论文使用 XeLaTeX 编译：

```powershell
cd paper/latex
latexmk -xelatex -interaction=nonstopmode main.tex
Copy-Item main.pdf ..\..\submission\final_paper.pdf -Force
```

LaTeX 临时文件不会纳入版本管理。若没有 `latexmk`，可参考 `paper/latex/README.md` 使用 `xelatex` 和 `bibtex` 手动编译。

## 审查说明

- 仓库保留原始数据、处理后数据、核心脚本、论文源文件、关键结果表图和最终提交材料。
- 预测结果均来自附件数据和项目脚本计算，不手工编造指标或图表。
- 天气、节假日、活动日等外部因素只解释为统计关联，不写成因果结论。
- 若课程或竞赛要求披露辅助工具使用情况，可参考 `submission/ai_usage_record.md`；若无此要求，提交论文和预测表即可。
