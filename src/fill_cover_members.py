from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = ROOT / "paper" / "latex" / "main.tex"
DEFAULT_CSV = ROOT / "paper" / "cover_member_info.csv"


MEMBER_BLOCK_PATTERN = re.compile(
    r"\\begin\{flushleft\}\s*"
    r"\\textbf\{小组每位成员信息：\}.*?"
    r"\\end\{flushleft\}",
    re.DOTALL,
)


def read_members(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = ["student_id", "name", "gender", "college", "class_name", "phone"]
    missing = [col for col in required if rows and col not in rows[0]]
    if missing:
        raise ValueError(f"{path} 缺少字段: {missing}")
    members = []
    for row in rows:
        cleaned = {col: (row.get(col) or "").strip() for col in required}
        if any(cleaned.values()):
            members.append(cleaned)
    if not members:
        raise ValueError(f"{path} 中没有填写任何成员信息。")
    if len(members) > 3:
        raise ValueError("封面模板当前仅预留 3 行成员信息，成员记录不得超过 3 位。")
    return members


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def member_block(members: list[dict[str, str]]) -> str:
    rows = []
    for member in members:
        values = [
            latex_escape(member["student_id"]),
            latex_escape(member["name"]),
            latex_escape(member["gender"]),
            latex_escape(member["college"]),
            latex_escape(member["class_name"]),
            latex_escape(member["phone"]),
        ]
        rows.append(" & ".join(values) + r" \\[1.1em]")
    while len(rows) < 3:
        rows.append(r" & & & & & \\[1.1em]")
    return "\n".join(
        [
            r"\begin{flushleft}",
            r"\textbf{小组每位成员信息：}",
            r"\vspace{0.5cm}",
            r"\begin{center}",
            r"{\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{p{2.0cm}p{1.8cm}p{0.9cm}p{3.4cm}p{2.6cm}p{2.8cm}}",
            r"\toprule",
            r"学号 & 姓名 & 性别 & 学院 & 班级 & 电话 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{center}",
            r"\end{flushleft}",
        ]
    )


def anonymous_block() -> str:
    return "\n".join(
        [
            r"\begin{center}",
            r"\textbf{匿名提交版本：封面不展示成员个人信息。}",
            r"\end{center}",
        ]
    )


def replace_block(text: str, replacement: str) -> str:
    new_text, count = MEMBER_BLOCK_PATTERN.subn(lambda _: replacement, text, count=1)
    if count != 1:
        raise RuntimeError("未能在 main.tex 中定位封面成员信息块。")
    return new_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill LaTeX cover member information.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="member info CSV path")
    parser.add_argument("--main-tex", type=Path, default=MAIN_TEX, help="main.tex path")
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="replace member table with an anonymous submission notice",
    )
    args = parser.parse_args()

    text = args.main_tex.read_text(encoding="utf-8")
    replacement = anonymous_block() if args.anonymous else member_block(read_members(args.csv))
    args.main_tex.write_text(replace_block(text, replacement), encoding="utf-8")
    print(f"updated {args.main_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
