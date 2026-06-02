"""Create a labeled overview image for the QED PaCMAP PNG sweep.

This is intentionally a small one-off utility for the current filenames, e.g.
nn020_mn0p25_fp12p0_qed_weighted_small.png.

Usage:
    .venv/bin/python scripts/arrange_panes.py
    .venv/bin/python scripts/arrange_panes.py --input pacmap_qed --out pacmap_qed/overview.png
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


FILENAME_RE = re.compile(r"nn(?P<nn>\d+)_mn(?P<mn>[0-9p]+)_fp(?P<fp>[0-9p]+)_qed_weighted")


@dataclass(frozen=True)
class Pane:
    path: Path
    n_neighbors: int
    mn_ratio: str
    fp_ratio: str


def _ratio_label(raw: str) -> str:
    return raw.replace("p", ".")


def _ratio_sort_key(raw: str) -> float:
    return float(_ratio_label(raw))


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)  # type: ignore
    return ImageFont.load_default()  # type: ignore


def parse_panes(input_dir: Path) -> list[Pane]:
    panes = []
    for path in sorted(input_dir.glob("*.png")):
        match = FILENAME_RE.search(path.name)
        if match is None:
            continue
        panes.append(
            Pane(
                path=path,
                n_neighbors=int(match.group("nn")),
                mn_ratio=match.group("mn"),
                fp_ratio=match.group("fp"),
            )
        )
    return panes


def make_overview(
    panes: list[Pane],
    *,
    out_path: Path,
    tile_width: int,
    padding: int,
) -> None:
    if not panes:
        raise ValueError("No matching QED PaCMAP PNGs found.")

    with Image.open(panes[0].path) as first:
        aspect = first.height / first.width
    tile_size = (tile_width, round(tile_width * aspect))

    n_values = sorted({pane.n_neighbors for pane in panes})
    mn_values = sorted({pane.mn_ratio for pane in panes}, key=_ratio_sort_key)
    fp_values = sorted({pane.fp_ratio for pane in panes}, key=_ratio_sort_key)
    by_key = {(pane.n_neighbors, pane.mn_ratio, pane.fp_ratio): pane for pane in panes}

    title_font = _font(34, bold=True)
    section_font = _font(26, bold=True)
    label_font = _font(18, bold=True)
    missing_font = _font(18)

    left_label_w = 92
    top_title_h = 78
    section_title_h = 44
    col_label_h = 34
    row_gap = padding
    section_h = (
        section_title_h
        + col_label_h
        + len(mn_values) * tile_size[1]
        + (len(mn_values) - 1) * row_gap
    )
    width = left_label_w + len(fp_values) * tile_size[0] + (len(fp_values) - 1) * padding
    height = top_title_h + len(n_values) * section_h + (len(n_values) - 1) * padding

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    title = f"PaCMAP QED sweep overview ({len(panes)} PNGs)"
    draw.text((0, 18), title, fill=(20, 20, 20), font=title_font)

    y = top_title_h
    for n_neighbors in n_values:
        draw.text((0, y), f"n_neighbors = {n_neighbors}", fill=(20, 20, 20), font=section_font)
        y += section_title_h

        for col, fp_ratio in enumerate(fp_values):
            x = left_label_w + col * (tile_size[0] + padding)
            label = f"FP {_ratio_label(fp_ratio)}"
            draw.text((x + 6, y + 5), label, fill=(40, 40, 40), font=label_font)
        y += col_label_h

        for row, mn_ratio in enumerate(mn_values):
            row_y = y + row * (tile_size[1] + row_gap)
            draw.text(
                (8, row_y + 8),
                f"MN\n{_ratio_label(mn_ratio)}",
                fill=(40, 40, 40),
                font=label_font,
                spacing=2,
            )
            for col, fp_ratio in enumerate(fp_values):
                x = left_label_w + col * (tile_size[0] + padding)
                pane = by_key.get((n_neighbors, mn_ratio, fp_ratio))
                if pane is None:
                    draw.rectangle(
                        [x, row_y, x + tile_size[0] - 1, row_y + tile_size[1] - 1],
                        fill=(245, 245, 245),
                        outline=(210, 210, 210),
                    )
                    draw.text(
                        (x + 16, row_y + 16),
                        "missing",
                        fill=(120, 120, 120),
                        font=missing_font,
                    )
                    continue

                with Image.open(pane.path) as image:
                    image = ImageOps.contain(image.convert("RGB"), tile_size)
                tile = Image.new("RGB", tile_size, "white")
                offset = ((tile_size[0] - image.width) // 2, (tile_size[1] - image.height) // 2)
                tile.paste(image, offset)
                canvas.paste(tile, (x, row_y))
                draw.rectangle(
                    [x, row_y, x + tile_size[0] - 1, row_y + tile_size[1] - 1],
                    outline=(225, 225, 225),
                )

        y += len(mn_values) * tile_size[1] + (len(mn_values) - 1) * row_gap + padding

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Wrote {out_path} ({canvas.width}x{canvas.height})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("pacmap_qed"))
    parser.add_argument("--out", type=Path, default=Path("pacmap_qed/overview.png"))
    parser.add_argument("--tile-width", type=int, default=300)
    parser.add_argument("--padding", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panes = parse_panes(args.input)
    make_overview(
        panes,
        out_path=args.out,
        tile_width=args.tile_width,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()
