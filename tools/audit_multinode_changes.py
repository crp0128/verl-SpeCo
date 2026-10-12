"""Classify actual Git changes against a fixed baseline, independent of IDE colors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(["git", "-c", "core.quotepath=false", *arguments], cwd=root, capture_output=True, check=True)
    return result.stdout


def audit(root: Path, baseline: str) -> dict:
    root = root.resolve()
    original = {s.decode("utf-8") for s in git(root, "ls-tree", "-rz", "--name-only", baseline).split(b"\0") if s}
    changed = {s.decode("utf-8") for s in git(root, "diff", baseline, "--name-only", "-z").split(b"\0") if s}
    candidates = {s.decode("utf-8") for s in git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0") if s}
    items = []
    for name in sorted(original | candidates):
        path = root / name
        if not path.is_file():
            status = "deleted" if name in original else "absent"
        elif name in original:
            status = "modified" if name in changed else "unchanged"
        else:
            status = "added"
        category = "tests" if name.startswith("tests/") else "code" if name.startswith("verl_speco/") else "docs" if name.startswith("docs/") else "tools" if name.startswith("tools/") else "project"
        items.append({"path": name, "status": status, "category": category,
                      **({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()} if path.is_file() else {})})
    return {"schema_version": 1, "baseline": baseline,
            "counts": {s: sum(i["status"] == s for i in items) for s in ("modified", "unchanged", "added", "deleted")}, "files": items}


def render(result: dict) -> str:
    counts = result["counts"]
    text = ["# 多机迁移实际文件分类", "", f"基准：`{result['baseline']}`。分类依据 Git 相对基准的内容差异，包含暂存、未暂存及新增文件，不采用 IDE 颜色。", "",
            "| 分类 | 文件数 |", "| --- | ---: |"]
    for status, label in (("modified", "原有文件已修改"), ("unchanged", "原有文件未修改"), ("added", "新增文件"), ("deleted", "原有文件删除")):
        text.append(f"| {label} | {counts[status]} |")
    text += ["", "环境 `.venv`、缓存 `.cache`、测试输出 `outputs` 和本机 `.idea` 设置均被忽略，不属于待提交的迁移代码。目录颜色可能由子文件的新增或修改聚合产生。", ""]
    for status, label in (("modified", "原有文件已修改"), ("added", "新增文件"), ("unchanged", "原有文件未修改"), ("deleted", "原有文件删除")):
        text += [f"## {label}", ""]
        for category in ("code", "tests", "docs", "tools", "project"):
            rows = [i for i in result["files"] if i["status"] == status and i["category"] == category]
            if rows:
                text += [f"### {category}", ""] + [f"- `{i['path']}`" for i in rows] + [""]
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="f868e84a8acc768c95ec6e764dfc3a8ec67ab0a1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # Reserve reports before the audit so they are included on the first run.
    markdown = root / "docs/multinode_file_classification.md"
    index = root / "docs/multinode_file_classification.json"
    for path in (markdown, index):
        if not path.exists():
            path.write_text("", encoding="utf-8")
    result = audit(root, args.baseline)
    # These reports contain their own inventory; self-hashes would be circular.
    for item in result["files"]:
        if item["path"] in {"docs/multinode_file_classification.md", "docs/multinode_file_classification.json"}:
            item.pop("sha256", None)
    markdown.write_text(render(result), encoding="utf-8", newline="\n")
    index.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
