"""Create public, synthetic fixtures for manual task-lifecycle checks."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare safe DeskPilot task demo fixtures")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "dataset/fixtures/reliability_demo")
    parser.add_argument("--pages", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.pages <= 80:
        parser.error("--pages must be between 1 and 80")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "source_a.md": "# 资料 A：任务状态\n\n任务排队后保存可恢复描述。成功节点保存结果与检查点。\n"
                       "等待审批不是任务成功；用户拒绝后不得执行外部提交。\n",
        "source_b.md": "# 资料 B：失败处理\n\n瞬时网络故障可以有限重试，认证错误不自动重试。\n"
                       "结果不确定的邮件不能自动重发。暂停和取消不等同于副作用回滚。\n",
        "attachment.txt": "Synthetic attachment version 1. No personal information.\n",
    }
    for name, content in files.items():
        destination = root / name
        if not destination.exists():
            destination.write_text(content, encoding="utf-8")
        print(destination)
    pdf = root / "long_fixture.pdf"
    if not pdf.exists():
        import fitz
        with fitz.open() as document:
            for number in range(1, args.pages + 1):
                page = document.new_page()
                page.insert_text((45, 60), f"DeskPilot reliability fixture - page {number}", fontsize=16)
                for row in range(12):
                    page.insert_text((45, 100 + row * 28), f"Batch checkpoint {number}: recover validated pages, never replay unknown effects.")
            document.save(pdf)
    print(pdf)
    print("Existing files were kept unchanged. These fixtures contain no credentials or personal documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
