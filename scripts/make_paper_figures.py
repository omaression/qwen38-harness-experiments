#!/usr/bin/env python3
"""Generate publication-grade IEEE vector PDF 1.4 figures for docs/paper/figures/.

    python scripts/make_paper_figures.py            write the three PDF figures
    python scripts/make_paper_figures.py --check    verify figures match byte-for-byte
    python scripts/make_paper_figures.py --root DIR read/write under DIR instead of checkout

Outputs:
    docs/paper/figures/fig-architecture.pdf   (504pt x 165pt, double-column)
    docs/paper/figures/fig-power-noise.pdf    (252pt x 195pt, single-column)
    docs/paper/figures/fig-ablations.pdf      (504pt x 195pt, double-column)

Standard library only (Python 3.9+ compatible). Produces deterministic PDF 1.4 bytes
with exact xref offsets, valid MediaBox, balanced q/Q and BT/ET operators, standard
Type1 fonts (Helvetica / Helvetica-Bold / Helvetica-Oblique / Symbol), and no timestamps.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = "scripts/make_paper_figures.py"
FIG_DIR = Path("docs/paper/figures")

# Colorblind-safe IEEE palette (RGB in 0..1)
RGB = Tuple[float, float, float]


def _hex(h: str) -> RGB:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


NAVY: RGB = _hex("#1b4965")
NAVY_LIGHT: RGB = _hex("#eef4f8")
NAVY_MID: RGB = _hex("#5c859e")
TEAL: RGB = _hex("#2a9d8f")
TEAL_LIGHT: RGB = _hex("#eaf6f4")
AMBER: RGB = _hex("#e9c46a")
AMBER_DEEP: RGB = _hex("#d97706")
AMBER_LIGHT: RGB = _hex("#fef7e0")
CORAL: RGB = _hex("#c1121f")
CORAL_LIGHT: RGB = _hex("#fdebec")
SLATE: RGB = _hex("#475569")
SLATE_LIGHT: RGB = _hex("#f1f5f9")
INK: RGB = _hex("#0f172a")
GRID: RGB = _hex("#e2e8f0")
BORDER: RGB = _hex("#cbd5e1")
WHITE: RGB = (1.0, 1.0, 1.0)


# Standard Adobe Helvetica character widths (1/1000 em) for accurate alignment
_HELV_WIDTHS: Dict[str, int] = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667, "'": 191,
    "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
    "0": 556, "1": 556, "2": 556, "3": 556, "4": 556, "5": 556, "6": 556, "7": 556,
    "8": 556, "9": 556, ":": 278, ";": 278, "<": 584, "=": 584, ">": 584, "?": 556,
    "@": 1015, "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778,
    "H": 722, "I": 278, "J": 500, "K": 667, "L": 556, "M": 833, "N": 722, "O": 778,
    "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556,
    "`": 333, "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556,
    "h": 556, "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556, "o": 556,
    "p": 556, "q": 556, "r": 333, "s": 500, "t": 278, "u": 556, "v": 500, "w": 722,
    "x": 500, "y": 500, "z": 500, "{": 334, "|": 260, "}": 334, "~": 584,
}

# Unicode-to-PDF Symbol font mappings (octal/char code, width in 1/1000 em)
_SYMBOL_MAP: Dict[str, Tuple[int, int]] = {
    "\u0394": (0x44, 612),   # Delta (D)
    "\u03b1": (0x61, 631),   # alpha (a)
    "\u2264": (0xA3, 549),   # <=
    "\u2265": (0xB3, 549),   # >=
    "\u2248": (0xBB, 549),   # approx
    "\u2192": (0xAE, 987),   # rightarrow
    "\u00d7": (0xB4, 549),   # times
    "\u00b1": (0xB1, 549),   # plusminus
    "\u2212": (0x2D, 549),   # minus
}

# WinAnsi single-byte extensions for Helvetica
_WINANSI_EXTRA: Dict[str, Tuple[int, int]] = {
    "\u2013": (0x96, 556),   # en-dash
    "\u2014": (0x97, 1000),  # em-dash
    "\u00b7": (0xB7, 278),   # middle dot
}


def _fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" or not s else s


def _rgb(c: RGB) -> str:
    return f"{_fmt(c[0])} {_fmt(c[1])} {_fmt(c[2])}"


def _escape_pdf_bytes(raw: bytes) -> str:
    out: List[str] = []
    for b in raw:
        if b == 0x28:
            out.append(r"\(")
        elif b == 0x29:
            out.append(r"\)")
        elif b == 0x5C:
            out.append(r"\\")
        elif 0x20 <= b <= 0x7E:
            out.append(chr(b))
        else:
            out.append(f"\\{b:03o}")
    return "".join(out)


def _split_runs(text: str, bold: bool = False, italic: bool = False) -> List[Tuple[str, bytes, int]]:
    """Split text into (font_key, raw_bytes, width_1000) runs across Helvetica and Symbol."""
    base_font = "F2" if bold else ("F3" if italic else "F1")
    bold_scale = 1.045 if bold else 1.0
    runs: List[List[object]] = []
    for ch in text:
        if ch in _SYMBOL_MAP:
            code, w = _SYMBOL_MAP[ch]
            font = "F4"
            width = w
        elif ch in _WINANSI_EXTRA:
            code, w = _WINANSI_EXTRA[ch]
            font = base_font
            width = int(round(w * bold_scale))
        else:
            code = ord(ch)
            if code > 255:
                raise ValueError(f"Unsupported character {ch!r} (U+{ord(ch):04X}) in PDF text {text!r}")
            font = base_font
            width = int(round(_HELV_WIDTHS.get(ch, 556) * bold_scale))
        if runs and runs[-1][0] == font:
            ba = runs[-1][1]
            assert isinstance(ba, bytearray)
            ba.append(code)
            runs[-1][2] = int(runs[-1][2]) + width  # type: ignore[arg-type]
        else:
            runs.append([font, bytearray([code]), width])
    return [(str(f), bytes(b), int(w)) for f, b, w in runs]  # type: ignore[arg-type]


def measure_text(text: str, size: float, bold: bool = False, italic: bool = False) -> float:
    """Measure string width in points."""
    runs = _split_runs(text, bold=bold, italic=italic)
    return sum(w for _, _, w in runs) * size / 1000.0


class PDFCanvas:
    """Deterministic single-page PDF 1.4 vector canvas."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self._ops: List[str] = []
        self._q_depth = 0
        self._bt_depth = 0
        self._alphas: Dict[float, str] = {1.0: "GS0"}

    def _gs_for_alpha(self, alpha: float) -> str:
        a = round(alpha, 3)
        if a not in self._alphas:
            self._alphas[a] = f"GS{len(self._alphas)}"
        return self._alphas[a]

    def save(self) -> None:
        self._ops.append("q")
        self._q_depth += 1

    def restore(self) -> None:
        assert self._q_depth > 0, "Unbalanced Q"
        self._ops.append("Q")
        self._q_depth -= 1

    def set_alpha(self, alpha: float) -> None:
        self._ops.append(f"/{self._gs_for_alpha(alpha)} gs")

    def set_stroke(self, color: RGB, width: float = 1.0, dash: Optional[Sequence[float]] = None) -> None:
        self._ops.append(f"{_rgb(color)} RG {_fmt(width)} w")
        if dash:
            arr = " ".join(_fmt(d) for d in dash)
            self._ops.append(f"[{arr}] 0 d")
        else:
            self._ops.append("[] 0 d")

    def set_fill(self, color: RGB) -> None:
        self._ops.append(f"{_rgb(color)} rg")

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: Optional[RGB] = None,
        stroke: Optional[RGB] = None,
        stroke_w: float = 0.75,
        dash: Optional[Sequence[float]] = None,
        alpha: float = 1.0,
    ) -> None:
        self.save()
        if alpha < 1.0:
            self.set_alpha(alpha)
        if fill is not None:
            self.set_fill(fill)
        if stroke is not None:
            self.set_stroke(stroke, stroke_w, dash=dash)
        self._ops.append(f"{_fmt(x)} {_fmt(y)} {_fmt(w)} {_fmt(h)} re")
        if fill is not None and stroke is not None:
            self._ops.append("B")
        elif fill is not None:
            self._ops.append("f")
        elif stroke is not None:
            self._ops.append("S")
        else:
            self._ops.append("n")
        self.restore()

    def rounded_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        r: float = 4.0,
        fill: Optional[RGB] = None,
        stroke: Optional[RGB] = None,
        stroke_w: float = 0.75,
        alpha: float = 1.0,
    ) -> None:
        r = min(r, w / 2.0, h / 2.0)
        k = 0.5522847498 * r
        self.save()
        if alpha < 1.0:
            self.set_alpha(alpha)
        if fill is not None:
            self.set_fill(fill)
        if stroke is not None:
            self.set_stroke(stroke, stroke_w)
        pts = [
            f"{_fmt(x + r)} {_fmt(y)} m",
            f"{_fmt(x + w - r)} {_fmt(y)} l",
            f"{_fmt(x + w - r + k)} {_fmt(y)} {_fmt(x + w)} {_fmt(y + r - k)} {_fmt(x + w)} {_fmt(y + r)} c",
            f"{_fmt(x + w)} {_fmt(y + h - r)} l",
            f"{_fmt(x + w)} {_fmt(y + h - r + k)} {_fmt(x + w - r + k)} {_fmt(y + h)} {_fmt(x + w - r)} {_fmt(y + h)} c",
            f"{_fmt(x + r)} {_fmt(y + h)} l",
            f"{_fmt(x + r - k)} {_fmt(y + h)} {_fmt(x)} {_fmt(y + h - r + k)} {_fmt(x)} {_fmt(y + h - r)} c",
            f"{_fmt(x)} {_fmt(y + r)} l",
            f"{_fmt(x)} {_fmt(y + r - k)} {_fmt(x + r - k)} {_fmt(y)} {_fmt(x + r)} {_fmt(y)} c",
            "h",
        ]
        self._ops.extend(pts)
        if fill is not None and stroke is not None:
            self._ops.append("B")
        elif fill is not None:
            self._ops.append("f")
        elif stroke is not None:
            self._ops.append("S")
        else:
            self._ops.append("n")
        self.restore()

    def top_rounded_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        r: float = 4.0,
        fill: Optional[RGB] = None,
    ) -> None:
        r = min(r, w / 2.0, h)
        k = 0.5522847498 * r
        self.save()
        if fill is not None:
            self.set_fill(fill)
        self._ops.extend([
            f"{_fmt(x)} {_fmt(y)} m",
            f"{_fmt(x + w)} {_fmt(y)} l",
            f"{_fmt(x + w)} {_fmt(y + h - r)} l",
            f"{_fmt(x + w)} {_fmt(y + h - r + k)} {_fmt(x + w - r + k)} {_fmt(y + h)} {_fmt(x + w - r)} {_fmt(y + h)} c",
            f"{_fmt(x + r)} {_fmt(y + h)} l",
            f"{_fmt(x + r - k)} {_fmt(y + h)} {_fmt(x)} {_fmt(y + h - r + k)} {_fmt(x)} {_fmt(y + h - r)} c",
            "h",
            "f",
        ])
        self.restore()

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: RGB = INK,
        width: float = 0.8,
        dash: Optional[Sequence[float]] = None,
        alpha: float = 1.0,
        cap: int = 1,
    ) -> None:
        self.save()
        if alpha < 1.0:
            self.set_alpha(alpha)
        self._ops.append(f"{cap} J")
        self.set_stroke(color, width, dash=dash)
        self._ops.append(f"{_fmt(x1)} {_fmt(y1)} m {_fmt(x2)} {_fmt(y2)} l S")
        self.restore()

    def polyline(
        self,
        points: Sequence[Tuple[float, float]],
        color: RGB = INK,
        width: float = 1.2,
        dash: Optional[Sequence[float]] = None,
    ) -> None:
        if len(points) < 2:
            return
        self.save()
        self._ops.append("1 J 1 j")
        self.set_stroke(color, width, dash=dash)
        self._ops.append(f"{_fmt(points[0][0])} {_fmt(points[0][1])} m")
        for px, py in points[1:]:
            self._ops.append(f"{_fmt(px)} {_fmt(py)} l")
        self._ops.append("S")
        self.restore()

    def polygon(
        self,
        points: Sequence[Tuple[float, float]],
        fill: Optional[RGB] = None,
        stroke: Optional[RGB] = None,
        stroke_w: float = 0.75,
        alpha: float = 1.0,
    ) -> None:
        if len(points) < 3:
            return
        self.save()
        if alpha < 1.0:
            self.set_alpha(alpha)
        if fill is not None:
            self.set_fill(fill)
        if stroke is not None:
            self.set_stroke(stroke, stroke_w)
        self._ops.append(f"{_fmt(points[0][0])} {_fmt(points[0][1])} m")
        for px, py in points[1:]:
            self._ops.append(f"{_fmt(px)} {_fmt(py)} l")
        self._ops.append("h")
        if fill is not None and stroke is not None:
            self._ops.append("B")
        elif fill is not None:
            self._ops.append("f")
        elif stroke is not None:
            self._ops.append("S")
        else:
            self._ops.append("n")
        self.restore()

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: Optional[RGB] = None,
        stroke: Optional[RGB] = None,
        stroke_w: float = 0.8,
    ) -> None:
        k = 0.5522847498 * r
        self.save()
        if fill is not None:
            self.set_fill(fill)
        if stroke is not None:
            self.set_stroke(stroke, stroke_w)
        self._ops.extend([
            f"{_fmt(cx + r)} {_fmt(cy)} m",
            f"{_fmt(cx + r)} {_fmt(cy + k)} {_fmt(cx + k)} {_fmt(cy + r)} {_fmt(cx)} {_fmt(cy + r)} c",
            f"{_fmt(cx - k)} {_fmt(cy + r)} {_fmt(cx - r)} {_fmt(cy + k)} {_fmt(cx - r)} {_fmt(cy)} c",
            f"{_fmt(cx - r)} {_fmt(cy - k)} {_fmt(cx - k)} {_fmt(cy - r)} {_fmt(cx)} {_fmt(cy - r)} c",
            f"{_fmt(cx + k)} {_fmt(cy - r)} {_fmt(cx + r)} {_fmt(cy - k)} {_fmt(cx + r)} {_fmt(cy)} c",
            "h",
        ])
        if fill is not None and stroke is not None:
            self._ops.append("B")
        elif fill is not None:
            self._ops.append("f")
        elif stroke is not None:
            self._ops.append("S")
        self.restore()

    def diamond(
        self,
        cx: float,
        cy: float,
        r: float,
        fill: Optional[RGB] = None,
        stroke: Optional[RGB] = None,
        stroke_w: float = 0.8,
    ) -> None:
        self.polygon([(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)], fill=fill, stroke=stroke, stroke_w=stroke_w)

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: RGB = SLATE,
        width: float = 1.0,
        head_len: float = 4.2,
        head_w: float = 2.6,
    ) -> None:
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        ux, uy = dx / dist, dy / dist
        nx, ny = -uy, ux
        bx = x2 - ux * head_len
        by = y2 - uy * head_len
        self.line(x1, y1, bx + ux * 0.4, by + uy * 0.4, color=color, width=width, cap=0)
        self.polygon(
            [(x2, y2), (bx + nx * head_w, by + ny * head_w), (bx - nx * head_w, by - ny * head_w)],
            fill=color,
        )

    def clip_rect(self, x: float, y: float, w: float, h: float) -> None:
        self.save()
        self._ops.append(f"{_fmt(x)} {_fmt(y)} {_fmt(w)} {_fmt(h)} re W n")

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float = 7.0,
        color: RGB = INK,
        bold: bool = False,
        italic: bool = False,
        align: str = "left",
        rotate_deg: float = 0.0,
    ) -> None:
        if not s:
            return
        runs = _split_runs(s, bold=bold, italic=italic)
        total_w = sum(w for _, _, w in runs) * size / 1000.0
        dx = 0.0
        if align == "center":
            dx = -total_w / 2.0
        elif align == "right":
            dx = -total_w
        self.save()
        self.set_fill(color)
        self._ops.append("BT")
        self._bt_depth += 1
        if abs(rotate_deg) > 1e-6:
            rad = math.radians(rotate_deg)
            c, sn = math.cos(rad), math.sin(rad)
            # Rotate around (x, y), then offset horizontally by dx along rotated axis
            rx = x + dx * c
            ry = y + dx * sn
            self._ops.append(f"{_fmt(c)} {_fmt(sn)} {_fmt(-sn)} {_fmt(c)} {_fmt(rx)} {_fmt(ry)} Tm")
        else:
            self._ops.append(f"1 0 0 1 {_fmt(x + dx)} {_fmt(y)} Tm")
        for font_key, raw_bytes, _w in runs:
            self._ops.append(f"/{font_key} {_fmt(size)} Tf")
            self._ops.append(f"({_escape_pdf_bytes(raw_bytes)}) Tj")
        self._ops.append("ET")
        self._bt_depth -= 1
        self.restore()

    def to_pdf_bytes(self) -> bytes:
        assert self._q_depth == 0, f"Unclosed graphics state q_depth={self._q_depth}"
        assert self._bt_depth == 0, f"Unclosed text state bt_depth={self._bt_depth}"
        stream_str = "\n".join(self._ops) + "\n"
        stream_bytes = stream_str.encode("latin-1")

        # Build objects deterministically
        objects: List[bytes] = []

        # 1: Catalog
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        # 2: Pages
        objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")

        # Alpha ExtGStates start at object 8
        gs_entries: List[str] = []
        gs_objs: List[bytes] = []
        next_id = 8
        for alpha_val, gs_name in sorted(self._alphas.items(), key=lambda kv: kv[1]):
            gs_entries.append(f"/{gs_name} {next_id} 0 R")
            a_str = _fmt(alpha_val)
            gs_objs.append(f"<< /Type /ExtGState /ca {a_str} /CA {a_str} >>".encode("ascii"))
            next_id += 1

        # 3: Page
        res = (
            f"<< /Font << /F1 4 0 R /F2 5 0 R /F3 6 0 R /F4 7 0 R >> "
            f"/ExtGState << {' '.join(gs_entries)} >> >>"
        )
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_fmt(self.width)} {_fmt(self.height)}] "
            f"/Resources {res} /Contents {next_id} 0 R >>"
        ).encode("ascii")
        objects.append(page_obj)

        # 4..7: Standard Type1 fonts
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique /Encoding /WinAnsiEncoding >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Symbol >>")

        # 8..: ExtGStates
        objects.extend(gs_objs)

        # Final: Contents stream
        content_obj = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"endstream"
        )
        objects.append(content_obj)

        buf = bytearray()
        buf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: List[int] = [0]
        for idx, body in enumerate(objects, start=1):
            offsets.append(len(buf))
            buf.extend(f"{idx} 0 obj\n".encode("ascii"))
            buf.extend(body)
            buf.extend(b"\nendobj\n")

        xref_offset = len(buf)
        n_objs = len(objects) + 1
        buf.extend(f"xref\n0 {n_objs}\n".encode("ascii"))
        buf.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

        buf.extend(
            f"trailer\n<< /Size {n_objs} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
        )
        return bytes(buf)


# =============================================================================
# Figure 1: Architecture & Dataflow (504pt x 165pt)
# =============================================================================

def draw_fig_architecture() -> bytes:
    W, H = 504.0, 165.0
    c = PDFCanvas(W, H)
    c.rect(0, 0, W, H, fill=WHITE)

    # Top title strip
    c.text(
        W / 2.0,
        H - 12.5,
        "llm-feature-eval-harness: Deterministic Offline Replay & 5-Gate Regression Pipeline",
        size=7.6,
        color=NAVY,
        bold=True,
        align="center",
    )
    c.line(8, H - 17.5, W - 8, H - 17.5, color=GRID, width=0.7)

    # 6 boxes across 504pt: left/right margin 6pt, 5 arrow gaps of 10pt -> box width 73.67pt
    margin_x = 6.0
    gap = 10.0
    n_boxes = 6
    box_w = (W - 2 * margin_x - (n_boxes - 1) * gap) / n_boxes  # 73.667pt
    box_y = 32.0
    box_h = 108.0
    hdr_h = 24.0

    stages = [
        (
            NAVY,
            NAVY_LIGHT,
            "1. Task Suite",
            None,
            [
                ("suite.json +", True),
                ("tasks.jsonl[.gz]", True),
                ("SHA-256 group", False),
                ("hash split:", False),
                ("dev / heldout", True),
                ("validate_suite()", False),
            ],
        ),
        (
            NAVY,
            NAVY_LIGHT,
            "2. Versioned",
            "Prompt",
            [
                ("bugfix/v2", True),
                ("judge/v2", True),
                ("review/v1", True),
                ("patch/v1", True),
                ("Pinned SHA-256", False),
                ("prompt templates", False),
            ],
        ),
        (
            TEAL,
            TEAL_LIGHT,
            "3. ReplayClient",
            "/ Live",
            [
                ("Canonical JSON", False),
                ("payload SHA-256", True),
                ("verified against", False),
                (".jsonl[.gz] tape", True),
                ("Refuses mismatch", False),
                ("Offline CI replay", False),
            ],
        ),
        (
            TEAL,
            TEAL_LIGHT,
            "4. Deterministic",
            "Verdicts",
            [
                ("exec_verdict sandbox", False),
                ("--recount diff_apply", False),
                ("+ AST + sast", True),
                ("(6 block / 6 flag)", False),
                ("Strict JSON &", False),
                ("think policy", False),
            ],
        ),
        (
            NAVY_MID,
            SLATE_LIGHT,
            "5. Paired",
            "Inference",
            [
                ("Cluster bootstrap", False),
                ("95% CI & SE", True),
                ("Exact McNemar p", True),
                ("MDE80 power floor", False),
                ("both_valid split", False),
                ("Paired item join", False),
            ],
        ),
        (
            CORAL,
            CORAL_LIGHT,
            "6. 5-Gate Rule",
            "& Ledger",
            [
                ("schema \u2192 think \u2192", False),
                ("SAST \u2192 primary \u2192", False),
                ("McNemar", True),
                ("Hash-chained", False),
                ("ledger.jsonl", True),
                ("Held-out budget", False),
            ],
        ),
    ]

    for i, (hdr_col, body_col, t1, t2, lines) in enumerate(stages):
        bx = margin_x + i * (box_w + gap)
        # Body background + border
        c.rounded_rect(bx, box_y, box_w, box_h, r=4.0, fill=body_col, stroke=BORDER, stroke_w=0.7)
        # Colored header
        c.top_rounded_rect(bx, box_y + box_h - hdr_h, box_w, hdr_h, r=4.0, fill=hdr_col)
        # Re-stroke outer border crisply
        c.rounded_rect(bx, box_y, box_w, box_h, r=4.0, fill=None, stroke=hdr_col, stroke_w=0.65)

        cx = bx + box_w / 2.0
        if t2 is None:
            c.text(cx, box_y + box_h - 14.5, t1, size=6.8, color=WHITE, bold=True, align="center")
        else:
            c.text(cx, box_y + box_h - 10.2, t1, size=6.6, color=WHITE, bold=True, align="center")
            c.text(cx, box_y + box_h - 19.2, t2, size=6.6, color=WHITE, bold=True, align="center")

        # Body lines
        start_y = box_y + box_h - hdr_h - 12.0
        step_y = 12.6
        for j, (txt, is_em) in enumerate(lines):
            ly = start_y - j * step_y
            # Subtle separator before last 2 lines in Box 6
            if i == 5 and j == 3:
                c.line(bx + 6.0, ly + 8.5, bx + box_w - 6.0, ly + 8.5, color=BORDER, width=0.5, dash=[2, 1.5])
            c.text(
                cx,
                ly,
                txt,
                size=5.85,
                color=INK if is_em else SLATE,
                bold=is_em,
                align="center",
            )

        # Connector arrow to next stage
        if i < n_boxes - 1:
            ax1 = bx + box_w + 1.0
            ax2 = bx + box_w + gap - 1.0
            ay = box_y + box_h / 2.0 - 4.0
            c.arrow(ax1, ay, ax2, ay, color=NAVY, width=1.1, head_len=3.8, head_w=2.3)

    # Bottom provenance / CI guarantee bar
    bar_y = 6.0
    bar_h = 19.0
    c.rounded_rect(margin_x, bar_y, W - 2 * margin_x, bar_h, r=3.0, fill=SLATE_LIGHT, stroke=BORDER, stroke_w=0.6)
    c.text(
        margin_x + 8.0,
        bar_y + 7.2,
        "Contract: Python stdlib only (0 third-party deps)  |  Shared typed records only",
        size=5.9,
        color=SLATE,
        bold=False,
    )
    c.text(
        W - margin_x - 8.0,
        bar_y + 7.2,
        "Output: Canonical JSON (no timestamps) \u2192 Byte-identical CI PROMOTE / REJECT",
        size=5.9,
        color=NAVY,
        bold=True,
        align="right",
    )

    return c.to_pdf_bytes()


# =============================================================================
# Figure 2: Power vs Serving-Noise Floor (252pt x 195pt)
# =============================================================================

def draw_fig_power_noise() -> bytes:
    W, H = 252.0, 195.0
    c = PDFCanvas(W, H)
    c.rect(0, 0, W, H, fill=WHITE)

    # Title
    c.text(
        W / 2.0,
        H - 11.5,
        "Detection Floor vs. Serving Noise Across Suite Sizes",
        size=7.6,
        color=NAVY,
        bold=True,
        align="center",
    )

    # Plot area
    px, py = 35.0, 29.0
    pw, ph = 207.0, 145.0
    n_min, n_max = 10.0, 450.0
    y_min, y_max = 0.0, 0.45

    def map_x(n: float) -> float:
        t = (math.log10(n) - math.log10(n_min)) / (math.log10(n_max) - math.log10(n_min))
        return px + t * pw

    def map_y(val: float) -> float:
        t = (val - y_min) / (y_max - y_min)
        return py + t * ph

    # Gridlines (horizontal every 0.05, vertical at log ticks)
    for k in range(0, 10):
        yv = k * 0.05
        yy = map_y(yv)
        is_major = (k % 2 == 0)
        c.line(px, yy, px + pw, yy, color=GRID, width=0.55 if is_major else 0.35)
        if is_major:
            c.text(px - 3.5, yy - 2.2, f"{yv:.2f}", size=6.0, color=SLATE, align="right")
    c.text(px - 3.5, map_y(0.45) - 2.2, "0.45", size=6.0, color=SLATE, align="right")

    x_ticks = [10, 20, 50, 100, 200, 400]
    for xt in x_ticks:
        xx = map_x(float(xt))
        c.line(xx, py, xx, py + ph, color=GRID, width=0.45)
        c.line(xx, py, xx, py - 2.5, color=SLATE, width=0.6)
        c.text(xx, py - 9.5, str(xt), size=6.0, color=SLATE, align="center")

    # Axis titles
    c.text(px + pw / 2.0, 7.0, "Paired suite size n (log scale)", size=6.6, color=INK, bold=True, align="center")
    c.text(10.0, py + ph / 2.0, "Effect size / detection threshold", size=6.6, color=INK, bold=True, align="center", rotate_deg=90.0)

    # Shaded C0 serving-noise band (0.00 to 0.041)
    y_noise = map_y(0.041)
    c.rect(px, py, pw, y_noise - py, fill=AMBER, alpha=0.28)
    c.line(px, y_noise, px + pw, y_noise, color=AMBER_DEEP, width=0.9, dash=[3.0, 2.0])
    c.text(
        px + 4.0,
        y_noise + 2.5,
        "C0 noise band (|\u0394| \u2264 0.041)",
        size=5.5,
        color=AMBER_DEEP,
        bold=True,
    )

    # Curve 1: McNemar discrete floor 6/n clipped to plot region
    c.clip_rect(px, py, pw, ph)
    curve_pts: List[Tuple[float, float]] = []
    steps = 120
    log_lo = math.log10(13.0)
    log_hi = math.log10(n_max)
    for idx in range(steps + 1):
        nn = 10.0 ** (log_lo + (log_hi - log_lo) * idx / steps)
        curve_pts.append((map_x(nn), map_y(6.0 / nn)))
    c.polyline(curve_pts, color=NAVY, width=1.6)
    c.restore()

    # Operating points
    # 1. demo: n=16, floor=0.375, MDE80 ~ 0.31
    x_demo = map_x(16.0)
    y_demo_fl = map_y(6.0 / 16.0)
    y_demo_mde = map_y(0.31)
    c.line(x_demo, y_demo_mde, x_demo, y_demo_fl, color=SLATE, width=0.6, dash=[1.5, 1.5])
    c.circle(x_demo, y_demo_fl, 2.4, fill=NAVY, stroke=WHITE, stroke_w=0.5)
    c.diamond(x_demo, y_demo_mde, 2.8, fill=TEAL, stroke=WHITE, stroke_w=0.5)

    # Label for demo (to the right of n=16, high up)
    lx_demo = map_x(23.5)
    ly_demo = map_y(0.36)
    c.line(x_demo + 2.5, (y_demo_fl + y_demo_mde) / 2.0, lx_demo - 2.0, ly_demo - 1.5, color=SLATE, width=0.55)
    c.text(lx_demo, ly_demo, "demo (n=16)", size=5.7, color=NAVY, bold=True)
    c.text(lx_demo, ly_demo - 7.5, "floor 0.375", size=5.4, color=INK)
    c.text(lx_demo, ly_demo - 14.5, "MDE80 \u2248 0.31", size=5.4, color=TEAL, bold=True)

    # 2. humanevalfix: n=74, floor=0.081, ceiling-compressed
    x_he = map_x(74.0)
    y_he = map_y(6.0 / 74.0)
    c.circle(x_he, y_he, 2.5, fill=NAVY, stroke=WHITE, stroke_w=0.5)
    # Place label cleanly in the open region below the 6/n curve (n=16..55, y=0.06..0.14)
    lx_he = map_x(14.5)
    ly_he = map_y(0.128)
    c.line(x_he - 2.5, y_he - 0.5, map_x(46.0), ly_he - 6.0, color=SLATE, width=0.55)
    c.text(lx_he, ly_he, "humanevalfix (n=74)", size=5.6, color=NAVY, bold=True)
    c.text(lx_he, ly_he - 7.3, "floor 0.081", size=5.3, color=INK)
    c.text(lx_he, ly_he - 14.3, "(ceiling-compressed)", size=5.2, color=CORAL)

    # 3. swe_coder_effort: n=169 held-out, Table XI MDE80=0.132-0.143 (D1=0.132)
    x_ce = map_x(169.0)
    y_ce_mde = map_y(0.132)
    y_ce_fl = map_y(6.0 / 169.0)
    c.circle(x_ce, y_ce_fl, 2.0, fill=NAVY, stroke=WHITE, stroke_w=0.45)
    c.diamond(x_ce, y_ce_mde, 2.9, fill=TEAL, stroke=WHITE, stroke_w=0.5)
    lx_ce = map_x(62.0)
    ly_ce = map_y(0.215)
    c.line(x_ce - 2.6, y_ce_mde + 1.5, lx_ce + 22.0, ly_ce - 15.5, color=SLATE, width=0.55)
    c.text(lx_ce, ly_ce, "swe_coder_effort", size=5.6, color=NAVY, bold=True, align="center")
    c.text(lx_ce, ly_ce - 7.3, "(n=169 held-out)", size=5.3, color=SLATE, align="center")
    c.text(lx_ce, ly_ce - 14.3, "MDE80 = 0.132\u20130.143", size=5.4, color=TEAL, bold=True, align="center")

    # 4. swe_review_bench: n=183 held-out / 400 all, pooled MDE80=0.067 on n=400
    x_rb_ho = map_x(183.0)
    y_rb_ho_fl = map_y(6.0 / 183.0)
    x_rb_all = map_x(400.0)
    y_rb_all_fl = map_y(6.0 / 400.0)
    y_rb_all_mde = map_y(0.067)
    c.circle(x_rb_ho, y_rb_ho_fl, 2.0, fill=WHITE, stroke=NAVY, stroke_w=0.75)
    c.circle(x_rb_all, y_rb_all_fl, 2.0, fill=NAVY, stroke=WHITE, stroke_w=0.45)
    c.diamond(x_rb_all, y_rb_all_mde, 2.9, fill=TEAL, stroke=WHITE, stroke_w=0.5)

    lx_rb = map_x(432.0)
    ly_rb = map_y(0.260)
    c.line(x_rb_all, y_rb_all_mde + 3.2, x_rb_all, ly_rb - 16.5, color=SLATE, width=0.55)
    c.text(lx_rb, ly_rb, "swe_review_bench", size=5.6, color=NAVY, bold=True, align="right")
    c.text(lx_rb, ly_rb - 7.3, "n=183 held-out / 400 all", size=5.3, color=SLATE, align="right")
    c.text(lx_rb, ly_rb - 14.3, "MDE80 = 0.067 (n=400)", size=5.4, color=TEAL, bold=True, align="right")

    # Plot border
    c.rect(px, py, pw, ph, fill=None, stroke=SLATE, stroke_w=0.75)

    # Legend box (top-right)
    leg_w, leg_h = 116.0, 38.0
    leg_x = px + pw - leg_w - 4.0
    leg_y = py + ph - leg_h - 4.0
    c.rounded_rect(leg_x, leg_y, leg_w, leg_h, r=2.5, fill=WHITE, stroke=BORDER, stroke_w=0.6)

    # Legend item 1: McNemar floor
    c.line(leg_x + 5.0, leg_y + 29.5, leg_x + 16.0, leg_y + 29.5, color=NAVY, width=1.5)
    c.circle(leg_x + 10.5, leg_y + 29.5, 2.0, fill=NAVY)
    c.text(leg_x + 20.0, leg_y + 27.5, "McNemar floor 6/n (\u03b1=0.05)", size=5.4, color=INK)

    # Legend item 2: MDE80
    c.diamond(leg_x + 10.5, leg_y + 18.5, 2.5, fill=TEAL)
    c.text(leg_x + 20.0, leg_y + 16.5, "Measured MDE80 (80% power)", size=5.4, color=INK)

    # Legend item 3: C0 noise band
    c.rect(leg_x + 5.5, leg_y + 5.5, 10.0, 5.5, fill=AMBER, alpha=0.45)
    c.line(leg_x + 5.5, leg_y + 11.0, leg_x + 15.5, leg_y + 11.0, color=AMBER_DEEP, width=0.8, dash=[2.0, 1.2])
    c.text(leg_x + 20.0, leg_y + 6.3, "C0 serving-noise band", size=5.4, color=INK)

    return c.to_pdf_bytes()


# =============================================================================
# Figure 3: Ablations (504pt x 195pt, two panels)
# =============================================================================

def draw_fig_ablations() -> bytes:
    W, H = 504.0, 195.0
    c = PDFCanvas(W, H)
    c.rect(0, 0, W, H, fill=WHITE)

    # Vertical panel divider
    c.line(251.0, 8.0, 251.0, H - 8.0, color=GRID, width=0.7)

    # -------------------------------------------------------------------------
    # Panel (a): Code Review LoRA & Budget-Forcing Confound (x = 0..248)
    # -------------------------------------------------------------------------
    c.text(
        134.0,
        H - 11.5,
        "(a) Code Review LoRA & Budget-Forcing Confound",
        size=7.4,
        color=NAVY,
        bold=True,
        align="center",
    )

    ax, ay = 34.0, 31.0
    aw, ah = 208.0, 128.0
    ymin_a, ymax_a = 0.40, 0.85

    def map_ya(v: float) -> float:
        return ay + ((v - ymin_a) / (ymax_a - ymin_a)) * ah

    # Gridlines (0.40 to 0.85 step 0.05)
    for k in range(10):
        yv = 0.40 + k * 0.05
        yy = map_ya(yv)
        is_major = (k % 2 == 0)
        c.line(ax, yy, ax + aw, yy, color=GRID, width=0.55 if is_major else 0.35)
        if is_major or k == 9:
            c.text(ax - 3.5, yy - 2.0, f"{yv:.2f}", size=5.8, color=SLATE, align="right")

    c.text(10.0, ay + ah / 2.0, "Balanced Accuracy", size=6.5, color=INK, bold=True, align="center", rotate_deg=90.0)

    # Dashed chance line at 0.50 (label placed in open right margin)
    y_chance = map_ya(0.50)
    c.line(ax, y_chance, ax + aw, y_chance, color=SLATE, width=0.9, dash=[3.5, 2.5])

    # Compact legend strip above plot (y = 163..175)
    leg_items_a = [
        (NAVY_MID, "Uncapped Base"),
        (TEAL, "Review LoRA"),
        (NAVY, "Capped Base"),
        (CORAL, "Security LoRA"),
    ]
    lx_cur = ax + 2.0
    for col, lbl in leg_items_a:
        c.rect(lx_cur, ay + ah + 6.0, 7.0, 6.0, fill=col)
        c.text(lx_cur + 9.5, ay + ah + 7.0, lbl, size=5.3, color=INK)
        lx_cur += 12.0 + measure_text(lbl, 5.3) + 4.5

    def draw_bar_a(
        cx: float,
        bw: float,
        val: float,
        color: RGB,
        ci: Optional[Tuple[float, float]] = None,
        label_inside: bool = False,
    ) -> None:
        y0 = ay
        y1 = map_ya(val)
        c.rect(cx - bw / 2.0, y0, bw, y1 - y0, fill=color, stroke=INK, stroke_w=0.35)
        if ci is not None:
            ylo = map_ya(ci[0])
            yhi = map_ya(ci[1])
            c.line(cx, ylo, cx, yhi, color=INK, width=0.85)
            c.line(cx - 2.6, ylo, cx + 2.6, ylo, color=INK, width=0.85)
            c.line(cx - 2.6, yhi, cx + 2.6, yhi, color=INK, width=0.85)
            lbl_y = yhi + 2.5
        else:
            lbl_y = y1 + 2.5
        if label_inside:
            c.text(cx, ay + 6.0, f"{val:.3f}", size=5.1, color=WHITE, bold=True, align="center")
        else:
            c.text(cx, lbl_y, f"{val:.3f}", size=5.3, color=INK, bold=True, align="center")

    bw_a = 18.0
    # Group 1: SWE-Review-Bench All (n=400)
    g1_x = ax + 31.0
    draw_bar_a(g1_x - 11.0, bw_a, 0.533, NAVY_MID, ci=(0.484, 0.580), label_inside=True)
    draw_bar_a(g1_x + 11.0, bw_a, 0.720, TEAL, ci=(0.669, 0.767), label_inside=True)
    by1 = map_ya(0.790)
    c.line(g1_x - 11.0, by1 - 3.0, g1_x - 11.0, by1, color=NAVY, width=0.7)
    c.line(g1_x - 11.0, by1, g1_x + 11.0, by1, color=NAVY, width=0.7)
    c.line(g1_x + 11.0, by1, g1_x + 11.0, by1 - 3.0, color=NAVY, width=0.7)
    c.text(g1_x, by1 + 2.5, "+0.188 (p=2.3e-13)", size=5.2, color=NAVY, bold=True, align="center")
    c.text(g1_x, ay - 9.5, "SWE-Review All", size=5.5, color=INK, bold=True, align="center")
    c.text(g1_x, ay - 16.5, "(n=400)", size=5.2, color=SLATE, align="center")

    # Group 2: SWE-Review-Bench Both-Valid (n=324)
    g2_x = ax + 88.0
    draw_bar_a(g2_x - 11.0, bw_a, 0.652, NAVY_MID, ci=(0.599, 0.706), label_inside=True)
    draw_bar_a(g2_x + 11.0, bw_a, 0.742, TEAL, ci=(0.689, 0.792), label_inside=True)
    by2 = map_ya(0.815)
    c.line(g2_x - 11.0, by2 - 3.0, g2_x - 11.0, by2, color=NAVY, width=0.7)
    c.line(g2_x - 11.0, by2, g2_x + 11.0, by2, color=NAVY, width=0.7)
    c.line(g2_x + 11.0, by2, g2_x + 11.0, by2 - 3.0, color=NAVY, width=0.7)
    c.text(g2_x, by2 + 2.5, "+0.089 (p=6.2e-4)", size=5.2, color=NAVY, bold=True, align="center")
    c.text(g2_x, ay - 9.5, "Both-Valid", size=5.5, color=INK, bold=True, align="center")
    c.text(g2_x, ay - 16.5, "(n=324)", size=5.2, color=SLATE, align="center")

    # Subtle separator before Security Confound
    c.line(ax + 118.0, ay, ax + 118.0, ay + ah, color=BORDER, width=0.65, dash=[2.5, 2.0])

    # Group 3: Security Confound (n=396)
    g3_x = ax + 153.0
    bw_3 = 15.0
    draw_bar_a(g3_x - 18.0, bw_3, 0.551, NAVY_MID, ci=None)
    draw_bar_a(g3_x, bw_3, 0.639, NAVY, ci=None)
    draw_bar_a(g3_x + 18.0, bw_3, 0.599, CORAL, ci=None)
    by3 = map_ya(0.698)
    c.line(g3_x, by3 - 3.0, g3_x, by3, color=CORAL, width=0.75)
    c.line(g3_x, by3, g3_x + 18.0, by3, color=CORAL, width=0.75)
    c.line(g3_x + 18.0, by3, g3_x + 18.0, by3 - 3.0, color=CORAL, width=0.75)
    c.text(g3_x + 9.0, by3 + 2.5, "-0.040 vs capped", size=5.2, color=CORAL, bold=True, align="center")
    c.text(g3_x, ay - 9.5, "Security Confound", size=5.5, color=INK, bold=True, align="center")
    c.text(g3_x, ay - 16.5, "(n=396)", size=5.2, color=SLATE, align="center")

    # Chance label cleanly placed in open right margin (x = ax + 180..206)
    c.text(ax + aw - 2.5, y_chance + 2.5, "Chance", size=5.1, color=SLATE, italic=True, align="right")
    c.text(ax + aw - 2.5, y_chance - 6.5, "(0.50)", size=5.1, color=SLATE, italic=True, align="right")

    c.rect(ax, ay, aw, ah, fill=None, stroke=SLATE, stroke_w=0.75)

    # -------------------------------------------------------------------------
    # Panel (b): Reasoning Effort & Agentic Acceptance Gate (x = 256..504)
    # -------------------------------------------------------------------------
    c.text(
        379.0,
        H - 11.5,
        "(b) Reasoning Effort & Agentic Acceptance Gate",
        size=7.4,
        color=NAVY,
        bold=True,
        align="center",
    )

    bx, by = 285.0, 31.0
    bw_plot, bh = 190.0, 128.0

    def map_yp(pct: float) -> float:
        return by + (pct / 100.0) * bh

    def map_yt(tok_k: float) -> float:
        return by + (tok_k / 100.0) * bh

    # Gridlines & Left Y (0..100%) / Right Y (0k..100k tokens)
    for k in range(6):
        pct = k * 20.0
        yy = map_yp(pct)
        c.line(bx, yy, bx + bw_plot, yy, color=GRID, width=0.55)
        c.text(bx - 3.0, yy - 2.0, f"{int(pct)}%", size=5.8, color=NAVY, align="right")
        c.text(bx + bw_plot + 3.0, yy - 2.0, f"{int(pct)}k", size=5.8, color=AMBER_DEEP, align="left")

    c.text(261.0, by + bh / 2.0, "Usable / Resolved (%)", size=6.4, color=NAVY, bold=True, align="center", rotate_deg=90.0)
    c.text(496.5, by + bh / 2.0, "Tokens (k / item or res.)", size=6.2, color=AMBER_DEEP, bold=True, align="center", rotate_deg=270.0)

    # Panel (b) legend strip above plot (evenly spaced)
    c.rect(bx + 2.0, by + bh + 6.0, 7.0, 6.0, fill=TEAL)
    c.text(bx + 11.5, by + bh + 7.0, "Target %", size=5.3, color=INK)
    c.rect(bx + 47.0, by + bh + 6.0, 7.0, 6.0, fill=NAVY_MID)
    c.text(bx + 56.5, by + bh + 7.0, "xhigh %", size=5.3, color=INK)
    c.rect(bx + 88.0, by + bh + 6.0, 7.0, 6.0, fill=CORAL)
    c.text(bx + 97.5, by + bh + 7.0, "Veto %", size=5.3, color=INK)
    c.rect(bx + 130.0, by + bh + 6.0, 7.0, 6.0, fill=AMBER, stroke=AMBER_DEEP, stroke_w=0.45)
    c.text(bx + 139.5, by + bh + 7.0, "Tokens (k)", size=5.3, color=INK)

    # 5 paired columns: 3 in swe_coder_effort (xhigh, medium, low), 2 in 5-Agent Loop (Auditor Veto, tests+SAST)
    items_b = [
        (bx + 20.0, 41.4, 24.5, NAVY_MID, "xhigh", "41.4%", "24.5k"),
        (bx + 55.0, 86.2, 4.9, TEAL, "medium", "86.2%", "4.9k"),
        (bx + 90.0, 86.2, 4.2, TEAL, "low", "86.2%", "4.2k"),
        (bx + 132.0, 60.0, 91.5, CORAL, "Auditor Veto", "60%", "91.5k"),
        (bx + 169.0, 68.0, 13.1, TEAL, "tests+SAST", "68%", "13.1k"),
    ]

    # Separator between swe_coder_effort and 5-Agent Loop
    sep_x = bx + 110.0
    c.line(sep_x, by, sep_x, by + bh, color=BORDER, width=0.7, dash=[2.5, 2.0])

    # Sub-study group headers
    c.text(bx + 55.0, by - 17.0, "swe_coder_effort (n=297)", size=5.4, color=NAVY, bold=True, align="center")
    c.text(bx + 150.5, by - 17.0, "5-Agent Loop (n=50 repos)", size=5.4, color=NAVY, bold=True, align="center")

    sub_w = 10.0
    for idx, (cx, rate_pct, tok_k, rcol, lbl, rstr, tstr) in enumerate(items_b):
        rx = cx - sub_w - 0.8
        tx = cx + 0.8
        ry1 = map_yp(rate_pct)
        ty1 = map_yt(tok_k)
        c.rect(rx, by, sub_w, ry1 - by, fill=rcol, stroke=INK, stroke_w=0.35)
        c.rect(tx, by, sub_w, ty1 - by, fill=AMBER, stroke=AMBER_DEEP, stroke_w=0.55)

        # Value labels cleanly above each sub-bar
        c.text(rx + sub_w / 2.0, ry1 + 2.2, rstr, size=5.2, color=INK, bold=True, align="center")
        tok_lbl_x = tx + sub_w / 2.0 + (1.5 if idx == 4 else 0.0)
        c.text(tok_lbl_x, ty1 + 2.2, tstr, size=5.0, color=AMBER_DEEP, bold=True, align="center")

        # X-tick label
        c.text(cx, by - 8.5, lbl, size=5.3, color=INK, bold=(idx in (1, 4)), align="center")

        # Extra callouts placed strictly in open whitespace with zero collisions
        if idx == 1:
            c.text(cx, ry1 + 9.8, "5.0\u00d7  tok cut", size=5.0, color=TEAL, bold=True, align="center")
        elif idx == 3:
            c.text(rx, ry1 + 17.5, "34.8%", size=4.9, color=CORAL, bold=True, align="center")
            c.text(rx, ry1 + 10.8, "false-rej.", size=4.8, color=CORAL, bold=True, align="center")
        elif idx == 4:
            c.text(cx - 2.5, ry1 + 10.2, "0 false accepts", size=5.0, color=TEAL, bold=True, align="center")
            c.text(tx + sub_w / 2.0 + 3.0, ty1 + 17.5, "7.0\u00d7", size=5.0, color=TEAL, bold=True, align="center")
            c.text(tx + sub_w / 2.0 + 3.0, ty1 + 11.0, "tok cut", size=4.8, color=TEAL, bold=True, align="center")

    c.rect(bx, by, bw_plot, bh, fill=None, stroke=SLATE, stroke_w=0.75)

    return c.to_pdf_bytes()


# =============================================================================
# CLI & Verification
# =============================================================================

FIGURES = {
    "fig-architecture.pdf": draw_fig_architecture,
    "fig-power-noise.pdf": draw_fig_power_noise,
    "fig-ablations.pdf": draw_fig_ablations,
}


def build_figures() -> Dict[str, bytes]:
    return {str((FIG_DIR / name).as_posix()): fn() for name, fn in FIGURES.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the tree instead of writing")
    ap.add_argument("--root", type=Path, default=ROOT, help="the repository root (default: this checkout)")
    args = ap.parse_args(argv)
    root = args.root.resolve()

    outputs = build_figures()
    fig_dir = root / FIG_DIR
    stale: List[str] = []
    if fig_dir.is_dir():
        stale = sorted(
            p.relative_to(root).as_posix()
            for p in fig_dir.glob("*.pdf")
            if p.relative_to(root).as_posix() not in outputs
        )

    if args.check:
        bad = [
            rel
            for rel, content in outputs.items()
            if not (root / rel).is_file() or (root / rel).read_bytes() != content
        ]
        for rel in bad:
            print(f"make_paper_figures: {rel} is out of date", file=sys.stderr)
        for rel in stale:
            print(f"make_paper_figures: {rel} is not generated by this script", file=sys.stderr)
        if bad or stale:
            print(f"make_paper_figures: run `python {SCRIPT}` and commit the result", file=sys.stderr)
            return 1
        print(f"make_paper_figures: {len(outputs)} figures up to date")
        return 0

    for rel in stale:
        (root / rel).unlink()
    for rel, content in outputs.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_bytes() != content:
            path.write_bytes(content)
            print(f"wrote {rel} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
