#!/usr/bin/env python3
"""Safe, offline installer for the Kamidso Codex skills pack.

Preserves an existing kamidso-tdd/SKILL.md and appends only a new,
idempotently marked section; never replaces existing new-skill folders.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

PACK = Path(__file__).resolve().parent
SKILLS = PACK / ".agents" / "skills"
TDD = "kamidso-tdd"
NEW = ("kamidso-test-runner", "kamidso-evidence-grounding", "kamidso-live-qualification")
BEGIN = "<!-- KAMIDSO_TDD_PACK_EXTENSION_START -->"
END = "<!-- KAMIDSO_TDD_PACK_EXTENSION_END -->"


def install(repo: Path, dry_run: bool = False, skip_tdd_extension: bool = False) -> None:
    repo = repo.expanduser().resolve()
    agents = repo / "AGENTS.md"
    if not agents.is_file() or "Kamidso" not in agents.read_text(encoding="utf-8-sig"):
        raise RuntimeError(f"Not a verified Kamidso project root (missing suitable AGENTS.md): {repo}")
    dest = repo / ".agents" / "skills"
    print(f"Target: {dest}")

    for name in NEW:
        src = SKILLS / name
        target = dest / name
        if target.exists():
            print(f"SKIP existing skill (never overwrite automatically): {target}")
            continue
        print(f"{'WOULD INSTALL' if dry_run else 'INSTALL'}: {name}")
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, target)

    if skip_tdd_extension:
        print("SKIP tdd extension (requested)")
        return

    source_tdd = (SKILLS / TDD / "SKILL.md").read_text(encoding="utf-8")
    if source_tdd.count(BEGIN) != 1 or source_tdd.count(END) != 1:
        raise RuntimeError("Malformed bundled TDD extension markers")
    addition = source_tdd[source_tdd.index(BEGIN):source_tdd.index(END) + len(END)]
    tdd_target = dest / TDD / "SKILL.md"
    if not tdd_target.exists():
        if tdd_target.parent.exists():
            raise RuntimeError(f"Found an existing TDD skill folder without SKILL.md: {tdd_target.parent}")
        print(f"{'WOULD INSTALL' if dry_run else 'INSTALL'}: {TDD} (no existing skill)")
        if not dry_run:
            tdd_target.parent.mkdir(parents=True)
            tdd_target.write_text(source_tdd, encoding="utf-8")
        return

    existing = tdd_target.read_text(encoding="utf-8-sig")
    if BEGIN in existing:
        if END not in existing:
            raise RuntimeError("Existing TDD extension marker is incomplete; manual review required")
        print("SKIP existing TDD extension (already installed)")
        return
    print(f"{'WOULD EXTEND' if dry_run else 'EXTEND'} existing {TDD}/SKILL.md without replacing its contents")
    if not dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = tdd_target.with_name(f"SKILL.md.pre-pack-{stamp}.bak")
        if backup.exists():
            raise RuntimeError(f"Backup already exists: {backup}")
        shutil.copy2(tdd_target, backup)
        tdd_target.write_text(existing.rstrip() + "\n\n" + addition + "\n", encoding="utf-8")
        print(f"Original TDD skill backed up at: {backup}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Path to existing Kamidso project root")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing files")
    parser.add_argument("--skip-tdd-extension", action="store_true", help="Do not modify existing kamidso-tdd")
    args = parser.parse_args()
    try:
        install(args.repo, args.dry_run, args.skip_tdd_extension)
    except (OSError, RuntimeError, UnicodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Done. Review git diff before committing; restart Codex to discover new skills.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
