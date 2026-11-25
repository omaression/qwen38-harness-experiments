"""Synchronization checks between paper LaTeX sources, figures, tables, and Markdown report sections."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper"
MAIN_TEX = PAPER_DIR / "main.tex"
REPORT_DIR = ROOT / "docs" / "report"

_HARDCODED_SECTION = re.compile(r"\bSection[~ ]+(?:VII|VI|IV|V|III|II)\b")
_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_GENERATED_BLOCK_RE = re.compile(
    r"<!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->\n(?P<body>.*?)<!-- END GENERATED: (?P=name) -->",
    re.DOTALL,
)


def _strip_tex_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        # Remove inline unescaped % comments.
        lines.append(re.sub(r"(?<!\\)%.*$", "", line))
    return "\n".join(lines)


def _abstract_word_count(tex_text: str) -> int:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_text, re.DOTALL)
    assert match is not None, "main.tex must contain an abstract environment"
    raw = match.group(1)
    # Strip TeX formatting wrappers while preserving visible prose words.
    plain = re.sub(r"\\text\{([^}]*)\}", r"\1", raw)
    plain = re.sub(r"\\texttt\{([^}]*)\}", r"\1", plain)
    plain = re.sub(r"\\times\b", "x", plain)
    plain = re.sub(r"\\to\b", "to", plain)
    plain = re.sub(r"\\pm\b", "+/-", plain)
    plain = plain.replace("---", " ").replace("--", "-")
    plain = re.sub(r"[{}$\\]", " ", plain)
    return len(plain.split())


def test_paper_figures_check_passes() -> None:
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_paper_figures.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr or res.stdout


def test_main_tex_inputs_and_graphics_resolve() -> None:
    body = _strip_tex_comments(MAIN_TEX.read_text(encoding="utf-8"))
    inputs = _INPUT_RE.findall(body)
    graphics = _GRAPHICS_RE.findall(body)
    assert inputs, "main.tex must \\input generated tables"
    assert graphics, "main.tex must \\includegraphics generated figures"
    for target in inputs:
        p = PAPER_DIR / target
        if not p.is_file():
            p = PAPER_DIR / f"{target}.tex"
        assert p.is_file(), f"Missing \\input target under docs/paper/: {target}"
    for target in graphics:
        p = PAPER_DIR / target
        assert p.is_file(), f"Missing \\includegraphics target under docs/paper/: {target}"


def test_main_tex_hygiene_and_cross_references() -> None:
    raw = MAIN_TEX.read_text(encoding="utf-8")
    body = _strip_tex_comments(raw)
    assert not _HARDCODED_SECTION.search(body), "main.tex contains hardcoded 'Section II..VII' instead of \\ref{...}"
    assert len(re.findall(r"\\ref\{sec:[^}]+\}", body)) >= 5, "main.tex must cross-reference sections with \\ref{sec:...}"
    for path in [MAIN_TEX, *sorted(REPORT_DIR.glob("*.md"))]:
        text = path.read_text(encoding="utf-8")
        assert "Anonymous" not in text, f"{path.relative_to(ROOT)} contains 'Anonymous'"
        assert "PLACEHOLDER" not in text, f"{path.relative_to(ROOT)} contains 'PLACEHOLDER'"
        assert not re.search(r"\bTODO\b", text), f"{path.relative_to(ROOT)} contains 'TODO'"


def test_abstract_word_count_and_required_tables() -> None:
    raw = MAIN_TEX.read_text(encoding="utf-8")
    words = _abstract_word_count(raw)
    assert 150 <= words <= 250, f"IEEE abstract word count {words} outside [150, 250]"

    body = _strip_tex_comments(raw)
    for eid in ("exp-a", "exp-b", "exp-c", "exp-c-demo", "exp-d"):
        for kind in ("gate", "arms"):
            assert f"tables/{eid}-{kind}" in body, f"main.tex missing \\input{{tables/{eid}-{kind}}}"
    for fig in ("fig-architecture.pdf", "fig-power-noise.pdf", "fig-ablations.pdf"):
        assert f"figures/{fig}" in body, f"main.tex missing figure {fig}"


def test_results_md_generated_blocks_present_and_in_sync() -> None:
    results_md = (REPORT_DIR / "06-results.md").read_text(encoding="utf-8")
    found = {m.group("name"): m.group("body").strip() for m in _GENERATED_BLOCK_RE.finditer(results_md)}
    for eid in ("exp-a", "exp-b", "exp-c", "exp-c-demo", "exp-d"):
        for kind in ("gate", "arms"):
            key = f"{eid}-{kind}"
            assert key in found, f"docs/report/06-results.md missing marker {key}"
            assert found[key], f"docs/report/06-results.md block {key} is empty"

    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_report_tables.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr or res.stdout
