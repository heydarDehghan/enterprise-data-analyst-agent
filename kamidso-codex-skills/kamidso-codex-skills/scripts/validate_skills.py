#!/usr/bin/env python3
"""Static validator for the packaged SKILL.md files (standard library only)."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / ".agents" / "skills"
EXPECTED = {
    "kamidso-tdd", "kamidso-test-runner",
    "kamidso-evidence-grounding", "kamidso-live-qualification",
}
NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
REF = re.compile(r"\]\((references/[^)]+)\)")


def main() -> int:
    actual = {p.name for p in ROOT.iterdir() if p.is_dir()}
    assert actual == EXPECTED, f"Expected skills {EXPECTED}, got {actual}"
    for dirname in sorted(actual):
        root = ROOT / dirname
        path = root / "SKILL.md"
        data = path.read_text(encoding="utf-8")
        parts = data.split("---\n", 2)
        assert len(parts) == 3 and parts[0] == "", f"Missing YAML frontmatter in {path}"
        metadata = {}
        for line in parts[1].strip().splitlines():
            assert ": " in line, (path, line)
            k, v = line.split(": ", 1)
            metadata[k] = v
        assert set(metadata) == {"name", "description"}, (path, metadata)
        assert NAME.fullmatch(metadata["name"]), (path, metadata["name"])
        assert metadata["name"] == dirname, (path, metadata["name"])
        assert metadata["description"].strip(), path
        assert not any(other.name.lower() == "skill.md" for other in root.rglob("*") if other.is_file() and other != path), f"Duplicate skill manifest in {root}"
        assert len(parts[2].strip()) > 200, f"Empty skill body: {path}"
        for ref in REF.findall(parts[2]):
            assert (root / ref).is_file(), f"Broken reference: {path} -> {ref}"
        print(f"VALID: {dirname} ({len(data.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1)
