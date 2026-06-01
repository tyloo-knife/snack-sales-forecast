# 封面成员信息填写模板

当前 `paper/latex/main.tex` 的封面成员表已按用户提供信息填写。若赛前需要修改成员信息或改为匿名提交，请按以下方式处理。

## 方式 A：非匿名提交

当前 3 位成员信息已写入 `paper/cover_member_info.csv`：

| 学号 | 姓名 | 性别 | 学院 | 班级 | 电话 |
|---|---|---|---|---|---|
| 25011076 | 吴渔桐 | 男 | 化学与分子工程学院 | 化学类(新工科)2503 | 18101855669 |
| 25012721 | 张哲凌 | 男 | 数学学院 | 2503 | 15026542486 |
| 25010409 | 朱饶杰 | 男 | 材料科学与工程学院 | 2514班 | 18101609530 |

## 方式 B：匿名提交

如果竞赛要求匿名提交，请确认封面是否应删除成员信息表，或按官方模板替换为队伍编号/参赛编号。

## 当前状态

模型优化、论文口径、预测表、提交包、封面成员信息和 PDF 编译均已完成。正式上传前仍需团队核对封面信息是否与报名系统一致。

## 工具化填写

已提供 CSV 模板：`paper/cover_member_info.csv`。

填写后运行：

```powershell
C:\ProgramData\anaconda3\python.exe src\fill_cover_members.py
```

如果竞赛要求匿名提交，运行：

```powershell
C:\ProgramData\anaconda3\python.exe src\fill_cover_members.py --anonymous
```

运行后需要重新编译 `paper/latex/main.tex`，并把新的 PDF 同步到 `submission/final_paper.pdf`。
