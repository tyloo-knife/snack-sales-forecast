# 最终 PDF 编译与审查报告

生成时间：2026-05-03

## 1. PDF 文件

- PDF 路径：`paper/final_paper.pdf`
- LaTeX 主文件：`paper/latex/main.tex`
- LaTeX 分章目录：`paper/latex/sections/`
- 编译说明：`paper/latex/README.md`
- PDF 页数：21 页
- 页面规格：A4

## 2. 编译方式

本次使用 XeLaTeX 编译，命令为：

```powershell
cd paper/latex
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
Copy-Item build/main.pdf ../final_paper.pdf -Force
```

编译成功。日志中无 LaTeX Error、无 Undefined references、无 Overfull box。仅有 1 条 Underfull hbox 提示，属于局部断行松散，不影响阅读和提交。

## 3. 内容检查

| 检查项 | 结果 |
|---|---|
| 是否使用 XeLaTeX | 是 |
| 中文是否正常显示 | 是，抽检首页、正文页、图表页和附录页未见乱码 |
| 公式是否正常显示 | 是，移动平均、相关系数、固定效应回归、WAPE 等公式正常 |
| 图表是否显示 | 是，16 张图均来自 `figures/` 且已编入 PDF |
| 图表编号是否连续 | 是，表 1--6、图 1--16 自动编号连续 |
| 表格是否溢出 | 未发现溢出；宽表已使用 `\resizebox{\textwidth}{!}{...}` |
| 页面是否有明显大面积空白 | 抽检未发现异常空白 |
| 参考文献是否正常 | 是，BibTeX 文献条目正常显示 |
| 附录是否正常 | 是，代码文件、结果文件和 AI 使用说明正常显示 |

## 4. 问题四口径检查

已检查 LaTeX 正文和 PDF 文本：

1. 严格 7 日递推验证为主验证口径；
2. 日滚动一步预测只作为辅助验证；
3. 综合 Ridge 在严格递推口径下具有竞争力；
4. 综合 Ridge 未超过问题一门店-商品简单指数平滑强 baseline；
5. 未出现“全局显著最优”或“显著提高未来 7 天预测精度”的夸大结论；
6. 未来天气、温度、风力和活动日均表述为历史同期参考情景，不是真实未来观测。

## 5. 整数化预测说明检查

论文中已说明：

1. `outputs/final_7day_forecast.csv` 是模型原始小数预测结果；
2. `outputs/final_7day_forecast_integer.csv` 是件数口径下的提交展示结果；
3. 整数化规则为预测值小于 0 时置为 0，其余四舍五入为整数；
4. 原始小数预测文件未覆盖。

## 6. 尚需人工确认事项

1. 正式提交是否使用 `paper/final_paper.pdf`，还是需要套学校或竞赛模板进一步排版。
2. 预测结果提交是否要求整数件数；若要求，使用 `outputs/final_7day_forecast_integer.csv`。
3. 是否需要把 AI 使用记录单独作为附件提交。
4. 是否需要从最终提交包中移除中间阶段报告和方法探索文件，仅保留论文、代码、最终图表和预测表。
