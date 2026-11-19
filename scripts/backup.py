"""备份数据库与检索索引，保留最近 N 份。

用法：python scripts/backup.py [--keep 7]
DuckDB 与索引文件都不大（几十 MB），整体打包最省事。
"""
import argparse
import os
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITEMS = [
    os.path.join(ROOT, "quant", "market.duckdb"),
    os.path.join(ROOT, "_distill", "rag", "children.jsonl"),
    os.path.join(ROOT, "_distill", "rag", "parents.jsonl"),
    os.path.join(ROOT, "_distill", "rag", "embeddings.npy"),
    os.path.join(ROOT, "_distill", "rag", "extra_children.jsonl"),
    os.path.join(ROOT, "_distill", "rag", "extra_parents.jsonl"),
]
OUT_DIR = os.path.join(ROOT, "backups")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = os.path.join(OUT_DIR, f"backup-{stamp}.zip")

    missing = [p for p in ITEMS if not os.path.exists(p)]
    if missing:
        print("警告：以下文件不存在，已跳过：")
        for m in missing:
            print("  " + m)

    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in ITEMS:
            if os.path.exists(p):
                zf.write(p, os.path.relpath(p, ROOT))
    size = os.path.getsize(target) / 1e6
    print(f"已备份 → {os.path.relpath(target, ROOT)}  ({size:.1f} MB)")

    old = sorted(f for f in os.listdir(OUT_DIR) if f.startswith("backup-") and f.endswith(".zip"))
    for f in old[:-args.keep]:
        os.remove(os.path.join(OUT_DIR, f))
        print(f"清理旧备份 {f}")


if __name__ == "__main__":
    main()
