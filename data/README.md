# 数据目录

## 目录说明

| 路径 | 内容 |
|---|---|
| `raw/` | 题目原文、历史零售明细、天气和日历附件 |
| `processed/` | 由清洗脚本生成的建模基础表 |

## 使用规则

- `raw/` 中的原始附件只读，不覆盖、不改名。
- `processed/` 中的数据可由脚本重新生成。
- 数据字段说明见项目根目录下的 `DATA_DICTIONARY.md`。

## 关键处理后数据

- `processed/daily_store_product_sales.csv`：日期-门店-商品粒度的日销量面板。
- `processed/modeling_base_table.csv`：合并天气、节假日、周末和活动日字段后的建模基础表。
