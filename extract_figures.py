#!/usr/bin/env python3
"""Crop the paper figures from DirCast_Alaya.pdf for the project page (300 dpi, white margins trimmed)."""
import numpy as np
import pymupdf
from PIL import Image
from pathlib import Path

PDF = Path(__file__).resolve().parents[2] / "DirCast_Alaya.pdf"
OUT = Path(__file__).resolve().parent / "static/img"
# name: (page, x0, y0, x1, y1) in PDF points (region above each caption)
FIGS = {"teaser": (0, 60, 382, 552, 604), "framework": (4, 60, 52, 552, 214),
        "scaling": (8, 348, 548, 552, 691), "qualitative": (9, 60, 52, 552, 341),
        "appendix1": (16, 60, 52, 552, 673), "appendix2": (17, 60, 52, 552, 673)}
doc = pymupdf.open(PDF)
for name, (pg, x0, y0, x1, y1) in FIGS.items():
    pix = doc[pg].get_pixmap(dpi=300, clip=pymupdf.Rect(x0, y0, x1, y1), alpha=False)
    im = Image.frombytes("RGB", (pix.w, pix.h), pix.samples)
    a = np.asarray(im); ys, xs = np.where((a < 245).any(2))
    im = im.crop((max(0, xs.min() - 8), max(0, ys.min() - 8), min(im.width, xs.max() + 9), min(im.height, ys.max() + 9)))
    if name == "scaling":
        im.save(OUT / f"{name}.png", optimize=True)
    else:
        im.save(OUT / f"{name}.jpg", quality=90, optimize=True, progressive=True)
    print(name, im.size)
