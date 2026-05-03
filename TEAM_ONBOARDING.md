# 队友上手文档

本文档面向没有参与前期建模的队友，目标是在 1 小时内看懂项目当前状态、知道从哪里读、怎么复现、接下来能接什么任务。

## 1. 项目目标

本仓库对应数学建模 A 题《休闲零食连锁店商品销量预测》。项目使用 `data/raw/` 中的历史销售明细、天气、节假日和活动日数据，完成休闲零食销量分析，并预测最后销售日期之后 7 天，即 `2022-04-01` 至 `2022-04-07` 的销量。

当前主数据口径是 `positive_sales`，表示顾客正向购买销量。负销量已按损耗或冲销类负向调整保留在审计字段中，不作为正常需求预测目标。最终预测文件为 `outputs/final_7day_forecast.csv`，但进入论文写作前仍需人工确认是否继续沿用该文件作为最终提交表。

## 2. 四个问题分别做了什么

| 问题 | 已完成内容 | 主要依据 |
|---|---|---|
| 问题一 | 分析不同门店、不同商品、门店-商品组合的历史销量差异；用移动平均、同星期均值、简单指数平滑预测未来 7 天销量 | `outputs/stage2_q1_modeling_report.md`、`tables/q1_*.csv`、`figures/q1_*.png` |
| 问题二 | 分析同一门店内商品销量相关性；用附件 `category` 字段整合同类零食；比较“单品预测后加总”和“类别聚合后直接预测” | `outputs/stage3_q2_relationship_report.md`、`tables/q2_*.csv`、`figures/q2_*.png` |
| 问题三 | 分析天气、温度、风力、节假日、周末、活动日与销量的统计关联；使用描述性统计、控制变量回归、随机森林置换重要性 | `outputs/stage4_q3_factor_analysis_report.md`、`outputs/q3_causality_language_review.md`、`tables/q3_*.csv` |
| 问题四 | 构造门店-商品粒度综合模型，加入历史滞后、滚动均值、日期、门店、商品、类别和外部变量；比较 Ridge、随机森林和 baseline；补充严格 7 日递推验证 | `outputs/stage5_q4_final_model_report.md`、`outputs/q4_information_leakage_review.md`、`outputs/q4_recursive_7day_model_comparison.md` |

方法探索的最终取舍建议在 `outputs/method_search/final_method_selection.md`。一句话概括：问题一保留原短期预测并补充 7 日窗口验证；问题二保留附件类别聚合；问题三建议把“合并天气 + 对数销量 + 历史控制固定效应回归”作为主分析；问题四用严格 7 日递推验证修正评价口径，不直接声称综合模型显著优于强 baseline。

## 3. 目录结构

| 路径 | 用途 | 注意事项 |
|---|---|---|
| `data/raw/` | 原始题目文档和附件 Excel | 只读，不得覆盖、删除或改名 |
| `data/processed/` | 清洗后的核心数据表 | 由清洗 Notebook 或脚本生成 |
| `notebooks/` | 分阶段分析 Notebook | 主流程按编号运行，`method_search/` 是方法探索 |
| `src/` | 可复用函数和阶段脚本 | 正式脚本主要是 `stage3`、`stage4`、`stage5` |
| `src/experimental/` | 方法探索脚本 | 不是正式主流程入口 |
| `figures/` | 论文和报告用图 | 按 `q1_` 到 `q4_` 前缀区分问题 |
| `tables/` | 指标、统计、验证、预测表 | 论文引用数值优先从这里核对 |
| `outputs/` | 阶段报告、最终预测、中间结论 | 新队友先读这里 |
| `outputs/method_search/` | 方法探索报告 | 用于进入论文前的方法取舍 |
| `paper/` | 论文分章节 Markdown 草稿 | 阶段 6 尚未完全完成 |
| `notes/` | 学习笔记 | 可补充概念解释 |
| `archive/` | 旧版本或废弃材料 | 不作为当前主结果 |
| `dist/` | 打包文件 | 不参与建模复现 |

根目录中还需要关注：

- `README.md`：当前项目导航和复现入口。
- `RESULT_LOG.md`：实验记录，记录每次重要建模结果和结论。
- `DATA_DICTIONARY.md`：字段说明和数据口径。
- `ASSUMPTIONS.md`：模型假设。
- `REVIEW_CHECKLIST.md`：论文提交前检查项。
- `AGENTS.md`：项目协作规则。

## 4. 如何配置环境

先进入项目根目录：

```powershell
cd "C:\Users\20215\Documents\CodexProjects\02-modeling\snack-sales-forecast"
```

推荐使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels openpyxl python-docx pillow jupyter nbconvert
```

如果 PowerShell 不允许激活虚拟环境，先在当前终端执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

当前仓库没有 `requirements.txt` 或 `environment.yml`，因此以上依赖需要手动安装。`xgboost`、`lightgbm`、`shap` 不是当前正式主流程必需依赖；环境检查中显示它们缺失时，不代表主流程失败。

检查环境：

```powershell
python --version
python -m pip show pandas numpy matplotlib seaborn scikit-learn scipy statsmodels openpyxl jupyter nbconvert
jupyter --version
```

启动 Notebook：

```powershell
jupyter lab
```

或：

```powershell
jupyter notebook
```

## 5. 如何按顺序运行 Notebook 或脚本

### 5.1 主流程 Notebook 顺序

从项目根目录启动 Jupyter 后，按下列顺序运行：

| 顺序 | Notebook | 作用 | 主要输出 |
|---:|---|---|---|
| 0 | `notebooks/00_environment_check.ipynb` | 检查 Python 和核心库 | 终端或 Notebook 输出 |
| 1 | `notebooks/01_data_inspection.ipynb` | 读取原始 Excel，检查表结构、缺失、重复、异常 | `outputs/stage1_data_inspection_report.md` |
| 2 | `notebooks/02_data_cleaning.ipynb` | 标准化字段，构造日销量面板，合并外部变量 | `data/processed/daily_store_product_sales.csv`、`data/processed/modeling_base_table.csv` |
| 3 | `notebooks/03_exploratory_analysis.ipynb` | 门店、商品、趋势、星期效应分析 | `figures/q1_*.png`、`tables/q1_*sales_stats.csv` |
| 4 | `notebooks/04_baseline_forecast.ipynb` | 问题一 baseline 和未来 7 天预测 | `outputs/stage2_q1_modeling_report.md`、`tables/q1_*.csv` |
| 5 | `notebooks/05_product_relationship_and_category_forecast.ipynb` | 问题二商品关联和类别预测结果查看 | `outputs/stage3_q2_relationship_report.md`、`tables/q2_*.csv` |
| 6 | `notebooks/06_external_factor_analysis.ipynb` | 问题三外部因素统计关联结果查看 | `outputs/stage4_q3_factor_analysis_report.md`、`tables/q3_*.csv` |
| 7 | `notebooks/07_final_prediction.ipynb` | 问题四综合模型结果、显著性检验和最终预测查看 | `outputs/stage5_q4_final_model_report.md`、`outputs/final_7day_forecast.csv` |

`notebooks/05_feature_engineering.ipynb` 和 `notebooks/06_model_comparison.ipynb` 是较早的辅助 Notebook，不是当前正式主流程必跑入口。

### 5.2 脚本复现入口

正式脚本主要覆盖问题二到问题四。运行前确保已经有 `data/processed/modeling_base_table.csv`：

```powershell
python src/stage3_q2_analysis.py
python src/stage4_q3_analysis.py
python src/stage5_q4_analysis.py
python src/stage5_q4_recursive_validation.py
```

脚本输出会写入 `tables/`、`figures/`、`outputs/`、`paper/`。如果只是检查结果，不要随便重跑；重跑可能覆盖当前报告、图表和表格。确实需要重跑时，先备份要覆盖的文件，例如：

```powershell
Copy-Item outputs\stage5_q4_final_model_report.md outputs\stage5_q4_final_model_report.md.manual-backup
python src/stage5_q4_analysis.py
```

### 5.3 方法探索入口

方法探索不是主流程，但论文定稿前需要阅读：

```powershell
Get-Content outputs\method_search\final_method_selection.md
```

对应 Notebook 和脚本：

- `notebooks/method_search/q1_method_search.ipynb`
- `notebooks/method_search/q2_method_search.ipynb`
- `notebooks/method_search/q3_method_search.ipynb`
- `notebooks/method_search/q4_method_search.ipynb`
- `src/experimental/q1_method_search.py`
- `src/experimental/q2_method_search.py`
- `src/experimental/q3_method_search.py`
- `src/experimental/q4_method_search.py`

这些文件用于比较更多方法，不建议新队友第一小时先跑。

## 6. 每个阶段的输入和输出

| 阶段 | 输入 | 操作 | 输出 |
|---|---|---|---|
| 阶段 0：题目理解 | `data/raw/A-休闲零食连锁店商品销量预测.docx`、两个附件 Excel | 明确四问任务、数据字段、总体方案 | `outputs/stage0_problem_understanding.md`、`PROJECT_PLAN.md`、`TODO.md` |
| 阶段 1：数据读取与预处理 | `data/raw/附件一：历史零售明细统计.xlsx`、`data/raw/附件二：某市近两年天气等数据.xlsx` | 检查表结构，标准化字段，聚合到日销量，合并外部变量 | `data/processed/daily_store_product_sales.csv`、`data/processed/modeling_base_table.csv`、`outputs/stage1_*.md` |
| 阶段 2：问题一 | `data/processed/daily_store_product_sales.csv` | 描述性统计，移动平均、同星期均值、指数平滑预测 | `outputs/stage2_q1_modeling_report.md`、`tables/q1_*.csv`、`figures/q1_*.png` |
| 阶段 3：问题二 | `data/processed/modeling_base_table.csv`、问题一预测结果 | 商品相关性矩阵，类别聚合，类别预测 | `outputs/stage3_q2_relationship_report.md`、`tables/q2_*.csv`、`figures/q2_*.png` |
| 阶段 4：问题三 | `data/processed/modeling_base_table.csv` | 外部因素描述性统计、固定效应回归、随机森林置换重要性 | `outputs/stage4_q3_factor_analysis_report.md`、`outputs/q3_causality_language_review.md`、`tables/q3_*.csv`、`figures/q3_*.png` |
| 阶段 5：问题四 | `data/processed/modeling_base_table.csv`、前三问结果 | 综合 Ridge/随机森林/baseline，对比误差，生成未来 7 天预测，做泄露检查和递推验证 | `outputs/stage5_q4_final_model_report.md`、`outputs/final_7day_forecast.csv`、`outputs/q4_information_leakage_review.md`、`outputs/q4_recursive_7day_model_comparison.md` |
| 阶段 6：完整论文 | `paper/` 草稿、`outputs/` 报告、`tables/` 表、`figures/` 图 | 整合成完整论文 | 当前尚未生成完整最终论文 |
| 阶段 7：答辩与检查 | 完整论文、代码、结果表、图表 | 查一致性、准备答辩问题 | `REVIEW_CHECKLIST.md` 已有，`outputs/final_review_checklist.md` 和 `outputs/defense_notes.md` 尚未完成 |

## 7. 常见错误

| 错误 | 现象 | 处理方式 |
|---|---|---|
| 没在项目根目录运行 | `ModuleNotFoundError: No module named 'src'` 或路径找不到 | 先执行 `cd "C:\Users\20215\Documents\CodexProjects\02-modeling\snack-sales-forecast"` |
| 原始附件缺失 | `FileNotFoundError` | 检查 `data/raw/` 是否有 `附件一：历史零售明细统计.xlsx`、`附件二：某市近两年天气等数据.xlsx` |
| 缺少 Excel 读取库 | 读取 `.xlsx` 失败 | 执行 `python -m pip install openpyxl` |
| 缺少统计或建模库 | `No module named 'statsmodels'`、`No module named 'sklearn'` | 执行 `python -m pip install scikit-learn statsmodels scipy` |
| `xgboost` 或 `lightgbm` 缺失 | 环境检查显示 `NOT INSTALLED` | 当前正式主流程不依赖它们；不要因此中断 |
| 误把 `daily_sales` 当预测目标 | 预测结果受负销量冲销影响 | 当前主预测目标是 `positive_sales` |
| 随机切分时间序列 | 指标过于乐观，论文逻辑错误 | 必须按时间切分或滚动验证，不能随机打乱 |
| 把相关性写成因果 | 例如写“活动日导致销量增加” | 改成“活动日与销量上升呈统计关联” |
| 未来天气被写成真实观测 | 问题四结论被质疑泄露 | 写明未来天气和活动日是历史同期参考情景，不是附件真实未来值 |
| 覆盖原始数据 | `data/raw/` 文件被改动 | 禁止覆盖；如已误改，立即停止并说明 |
| 重跑脚本覆盖报告 | 阶段报告或结果表时间戳变化 | 重跑前先 `Copy-Item` 备份目标文件 |
| 忽略 `2022-03-31` 外部变量缺失 | 问题三/四验证日期对不上 | `附件二` 覆盖到 `2022-03-30`，综合模型验证通常不纳入 `2022-03-31` |

## 8. 队友分工建议

建议按“复现、问题建模、论文、审查”拆分，不要多人同时改同一个文件。

| 角色 | 负责范围 | 重点文件 |
|---|---|---|
| 数据与复现负责人 | 从 `00` 到 `02` Notebook 复跑，确认原始数据、清洗表和数据字典一致 | `data/raw/`、`data/processed/`、`DATA_DICTIONARY.md`、`outputs/stage1_*.md` |
| 问题一/二负责人 | 复核短期预测、商品关联、类别聚合；确认图表和表格能支撑论文结论 | `outputs/stage2_q1_modeling_report.md`、`outputs/stage3_q2_relationship_report.md`、`tables/q1_*.csv`、`tables/q2_*.csv` |
| 问题三/四负责人 | 复核外部因素措辞、信息泄露检查、严格 7 日递推验证和最终预测文件 | `outputs/stage4_q3_factor_analysis_report.md`、`outputs/q4_information_leakage_review.md`、`outputs/q4_recursive_7day_model_comparison.md`、`outputs/final_7day_forecast.csv` |
| 论文负责人 | 把阶段报告、表格、图表整合为正式论文，统一符号、公式、图表编号 | `paper/`、`ASSUMPTIONS.md`、`REVIEW_CHECKLIST.md` |
| 终审负责人 | 检查结果是否可复现、是否有伪造字段、是否有因果夸大、提交格式是否满足题目 | `RESULT_LOG.md`、`outputs/method_search/final_method_selection.md`、`tables/`、`figures/` |

第一小时建议阅读顺序：

1. `README.md`
2. `TEAM_ONBOARDING.md`
3. `outputs/method_search/final_method_selection.md`
4. 自己负责问题对应的阶段报告
5. 对应的 `tables/q*_*.csv` 和 `figures/q*_*.png`

## 9. 提交前检查清单

### 9.1 代码和数据

- [ ] `data/raw/` 没有被修改、覆盖或删除。
- [ ] `data/processed/daily_store_product_sales.csv` 和 `data/processed/modeling_base_table.csv` 存在。
- [ ] `outputs/final_7day_forecast.csv` 存在，并确认是否作为最终提交预测表。
- [ ] 所有正式结论能在 `outputs/`、`tables/` 或 `figures/` 中找到依据。
- [ ] `RESULT_LOG.md` 记录了重要实验和方法取舍。

可用命令：

```powershell
git status --short
Get-ChildItem data\raw
Get-ChildItem data\processed
Get-ChildItem outputs\final_7day_forecast.csv
```

### 9.2 模型和指标

- [ ] 每个预测问题至少有 baseline 比较。
- [ ] 时间序列验证没有随机打乱。
- [ ] 指标包含 MAE、RMSE、WAPE。
- [ ] 问题四正文区分“日滚动一步预测”和“严格 7 日递推预测”。
- [ ] 如果写“显著改进”，必须对应显著性检验结果；否则写“数值上下降”或“未能证明显著改进”。

### 9.3 论文文字

- [ ] 不把相关性写成因果性。
- [ ] 不写数据里没有的字段，例如湿度。
- [ ] 每张图说明展示内容、支撑结论、对模型的启发。
- [ ] 每张表说明字段含义、主要结论、与题目问题的关系。
- [ ] 问题三统一使用“统计关联”“条件关联”等表述。
- [ ] 问题四说明未来天气和活动日属于情景假设，不是真实未来观测。

### 9.4 结果一致性

- [ ] 论文中的 WAPE、MAE、RMSE 与 `tables/` 中对应表一致。
- [ ] 论文中的最终预测日期是 `2022-04-01` 至 `2022-04-07`。
- [ ] 论文中的门店数、商品数、类别数与阶段 1 报告一致：7 家门店、12 个商品代码、13 个商品名称、8 个类别。
- [ ] 问题二“同类零食”使用附件 `category` 字段，不用销量规模分组替代正式类别。
- [ ] 问题四不直接声称综合模型严格优于所有候选模型。

### 9.5 最后建议

提交前先人工确认以下 5 件事：

1. 最终提交要求是每日预测还是未来 7 天总销量。
2. 是否继续使用 `positive_sales` 作为主预测目标。
3. `outputs/final_7day_forecast.csv` 是否保留为最终预测文件。
4. 预测销量是否需要整数化，若需要采用什么四舍五入规则。
5. 是否允许问题四使用历史同期天气和活动日情景。
