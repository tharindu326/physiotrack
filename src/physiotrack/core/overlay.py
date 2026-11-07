"""High-quality overlay rendering toolkit (core).

OpenCV's ``cv2.putText`` draws chunky Hershey *stroke* fonts, and ``cv2.line`` /
``polylines`` / ``rectangle`` render thin, hard-aliased primitives. Composited onto
video that reads as "pixelated / low quality" regardless of the video's resolution.

``OverlayCanvas`` fixes that at the source: every overlay element is drawn with
**supersampled anti-aliasing (SSAA)** and real **TrueType text** (Pillow), then the
canvas is downsampled with ``INTER_AREA``. The result composites onto a frame at the
*same footprint* as before -- so panel layouts are unchanged -- but text, plots,
gauges and arcs come out crisp and smooth instead of jagged.

All plotters and views draw through this class so quality is uniform and
resolution-independent: callers give coordinates and sizes in *display* space (the
final composited pixels); the canvas works internally at ``SS x`` that and downsamples
on :meth:`render`. Colors are given as OpenCV-style **BGR** or **BGRA** tuples to match
the rest of the codebase.

Example
-------
>>> ov = OverlayCanvas(460, 170, bg=(24, 22, 28), bg_alpha=0.55, border=(105, 95, 90))
>>> ov.text((10, 8), "HR  72 bpm", size=20, color=(235, 235, 235), bold=True)
>>> ov.polyline(pts, color=(120, 220, 120), width=2)
>>> panel_bgra = ov.render()          # (170, 460, 4) uint8, crisp
"""

import os
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Internal supersample factor. Everything is drawn this many times larger and then
# area-averaged down, which is what turns Pillow's (otherwise aliased) lines/shapes
# and the TrueType glyphs into smooth, high-quality edges. 3x is the sweet spot
# between quality and the cost of drawing on a 9x-area buffer.
SS = 3

_ASSET_FONTS = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts")
# Regular / bold TrueType faces, tried in order. The bundled DejaVu faces ship with
# the package so rendering is identical on every platform; system fonts and Pillow's
# bitmap default are only fallbacks.
_FONT_CANDIDATES = {
    False: [os.path.join(_ASSET_FONTS, "DejaVuSans.ttf"), "DejaVuSans.ttf", "arial.ttf"],
    True: [os.path.join(_ASSET_FONTS, "DejaVuSans-Bold.ttf"), "DejaVuSans-Bold.ttf", "arialbd.ttf"],
}


@lru_cache(maxsize=256)
def _font(px: int, bold: bool) -> ImageFont.FreeTypeFont:
    """Load a TrueType face at ``px`` pixels (cached). Falls back to Pillow's default."""
    px = max(1, int(px))
    for path in _FONT_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(path, px)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _rgba(color: Sequence[int]) -> Tuple[int, int, int, int]:
    """OpenCV-style BGR/BGRA -> Pillow RGBA."""
    if len(color) == 4:
        b, g, r, a = color
    else:
        b, g, r = color
        a = 255
    return int(r), int(g), int(b), int(a)


class OverlayCanvas:
    """A supersampled, TrueType-capable drawing surface that renders to a BGR(A) array.

    Parameters
    ----------
    width, height : int
        Display-space size (the footprint the canvas will occupy on the frame).
    alpha : bool
        If True (default) :meth:`render` returns BGRA for alpha compositing; else BGR.
    bg, bg_alpha : optional
        Fill the whole canvas with this BGR background at ``bg_alpha`` opacity.
    border : optional
        Draw a 1px (display-space) border in this BGR color around the canvas.
    radius : int
        Corner radius (display px) for the background/border, for a softer panel look.
    ss : int
        Supersample factor (defaults to the module-level :data:`SS`).
    """

    def __init__(self, width: int, height: int, *, alpha: bool = True,
                 bg: Optional[Sequence[int]] = None, bg_alpha: float = 0.55,
                 border: Optional[Sequence[int]] = None, radius: int = 0, ss: int = SS):
        self.w, self.h = int(width), int(height)
        self.alpha = bool(alpha)
        self.ss = max(1, int(ss))
        self._img = Image.new("RGBA", (self.w * self.ss, self.h * self.ss), (0, 0, 0, 0))
        self._d = ImageDraw.Draw(self._img)
        if bg is not None:
            a = int(np.clip(bg_alpha, 0.0, 1.0) * 255)
            self.rect((0, 0), (self.w - 1, self.h - 1), color=(*bg[:3], a), fill=True, radius=radius)
        if border is not None:
            self.rect((0, 0), (self.w - 1, self.h - 1), color=border, width=1, radius=radius)

    # -- scaling helpers (display space -> supersampled space) --
    def _s(self, v):
        return int(round(v * self.ss))

    def _pt(self, xy):
        return (self._s(xy[0]), self._s(xy[1]))

    # -- text --
    def text(self, xy, text: str, *, size: float, color=(235, 235, 235),
             bold: bool = False, anchor: str = "lt") -> None:
        """Draw crisp TrueType text. ``xy`` is display-space; ``size`` is display px.

        ``anchor`` follows Pillow (default 'lt' = left/top of the text box).
        """
        self._d.text(self._pt(xy), str(text), font=_font(self._s(size), bold),
                     fill=_rgba(color), anchor=anchor)

    def measure(self, text: str, size: float, bold: bool = False) -> Tuple[float, float]:
        """Text (width, height) in display-space px -- for right/center alignment."""
        l, t, r, b = self._d.textbbox((0, 0), str(text), font=_font(self._s(size), bold))
        return (r - l) / self.ss, (b - t) / self.ss

    # -- primitives (all sizes in display space) --
    def line(self, p1, p2, color, width: float = 1.0) -> None:
        self._d.line([self._pt(p1), self._pt(p2)], fill=_rgba(color),
                     width=max(1, self._s(width)))

    def polyline(self, pts, color, width: float = 1.0) -> None:
        if len(pts) < 2:
            return
        self._d.line([self._pt(p) for p in pts], fill=_rgba(color),
                     width=max(1, self._s(width)), joint="curve")

    def rect(self, p1, p2, color, *, fill: bool = False, width: float = 1.0,
             radius: int = 0) -> None:
        box = [self._pt(p1), self._pt(p2)]
        col = _rgba(color)
        if radius > 0:
            self._d.rounded_rectangle(box, radius=self._s(radius),
                                      fill=col if fill else None,
                                      outline=None if fill else col,
                                      width=max(1, self._s(width)))
        elif fill:
            self._d.rectangle(box, fill=col)
        else:
            self._d.rectangle(box, outline=col, width=max(1, self._s(width)))

    def circle(self, center, radius: float, color, *, fill: bool = True,
               width: float = 1.0) -> None:
        cx, cy = self._pt(center)
        r = self._s(radius)
        box = [cx - r, cy - r, cx + r, cy + r]
        col = _rgba(color)
        self._d.ellipse(box, fill=col if fill else None,
                        outline=None if fill else col,
                        width=max(1, self._s(width)))

    def arc(self, center, radius: float, start_deg: float, end_deg: float, color,
            width: float = 2.0) -> None:
        cx, cy = self._pt(center)
        r = self._s(radius)
        self._d.arc([cx - r, cy - r, cx + r, cy + r], start_deg, end_deg,
                    fill=_rgba(color), width=max(1, self._s(width)))

    def paste_bgr(self, img_bgr: np.ndarray, xy=(0, 0)) -> None:
        """Paste a plain BGR(A) image (e.g. a colorized map) into the canvas, opaque.

        The image is treated as display-space pixels and upsampled to the internal
        resolution with high-quality interpolation so it stays sharp.
        """
        x, y = self._pt(xy)
        h, w = img_bgr.shape[:2]
        big = cv2.resize(img_bgr, (self._s(w), self._s(h)), interpolation=cv2.INTER_CUBIC)
        if big.shape[2] == 3:
            big = cv2.cvtColor(big, cv2.COLOR_BGR2BGRA)
        rgba = big[:, :, [2, 1, 0, 3]]
        self._img.paste(Image.fromarray(rgba, "RGBA"), (x, y))

    # -- finish --
    def render(self) -> np.ndarray:
        """Downsample (area-average) to the display footprint and return a BGR(A) array."""
        rgba = np.asarray(self._img).astype(np.float32)      # (H*ss, W*ss, 4) RGBA
        # Premultiply by alpha before area-averaging so semi-transparent AA edges do
        # not bleed the (black) transparent background into the color channels.
        a = rgba[:, :, 3:4] / 255.0
        rgba[:, :, :3] *= a
        small = cv2.resize(rgba, (self.w, self.h), interpolation=cv2.INTER_AREA)
        sa = small[:, :, 3:4]
        np.divide(small[:, :, :3] * 255.0, np.maximum(sa, 1e-6), out=small[:, :, :3])
        small = np.clip(small, 0, 255).astype(np.uint8)
        bgra = small[:, :, [2, 1, 0, 3]]                     # RGBA -> BGRA
        return bgra if self.alpha else bgra[:, :, :3]


def draw_label(frame: np.ndarray, xy, text: str, *, size: float,
               color=(255, 255, 255), bold: bool = False,
               bg: Optional[Sequence[int]] = None, bg_alpha: float = 1.0,
               pad: int = 3, radius: int = 3) -> None:
    """Draw crisp TrueType text directly onto a BGR ``frame`` at top-left ``xy``.

    Renders into a *small* supersampled canvas sized to the text (optionally with a
    filled background box for legibility) and alpha-composites it -- so only a tiny
    region is supersampled, making it cheap enough for per-object, per-frame labels
    (bounding-box ids, etc.) without supersampling the whole frame.
    """
    text = str(text)
    probe = OverlayCanvas(1, 1)
    tw, th = probe.measure(text, size, bold)
    w, h = int(tw + 2 * pad), int(th + 2 * pad)
    ov = OverlayCanvas(w, h, bg=bg, bg_alpha=bg_alpha,
                       radius=radius if bg is not None else 0)
    ov.text((pad, pad), text, size=size, color=color, bold=bold)
    alpha_composite(frame, ov.render(), int(xy[0]), int(xy[1]))


def draw_info_panel(frame: np.ndarray, lines: Sequence[str], *,
                    corner: str = "top_left") -> np.ndarray:
    """Return a copy of ``frame`` with a semi-transparent text panel in one corner.

    The panel scales with the frame resolution, so the same call is legible on a
    phone clip and a 4K recording. Used by the runnable examples to stamp run
    context (subject counts, model name, device) onto saved media; reuse it
    anywhere a frame needs a few lines of status text.

    Args:
        frame (np.ndarray): BGR image ``(H, W, 3)``. Not modified.
        lines (Sequence[str]): The text lines, drawn top to bottom.
        corner (str, optional): ``"top_left"``, ``"top_right"``, ``"bottom_left"``
            or ``"bottom_right"``. Defaults to ``"top_left"``.

    Returns:
        np.ndarray: A new annotated BGR image with the same shape as ``frame``.

    Example:
        ```python
        from physiotrack.core.overlay import draw_info_panel

        annotated = draw_info_panel(annotated, [
            f"Faces detected: {len(result)}",
            f"Detector: {model_name}",
        ])
        ```
    """
    lines = [str(line) for line in lines]
    if not lines:
        return frame.copy()
    height, width = frame.shape[:2]
    size = float(np.clip(min(width, height) / 42.0, 14.0, 30.0))
    pad = int(round(size * 0.6))
    gap = int(round(size * 0.45))

    probe = OverlayCanvas(1, 1)
    measured = [probe.measure(line, size, bold=True) for line in lines]
    text_w = max(w for w, _ in measured)
    line_h = max(h for _, h in measured)
    panel_w = min(width, int(text_w + 2 * pad))
    panel_h = min(height, int(len(lines) * line_h + (len(lines) - 1) * gap + 2 * pad))

    ov = OverlayCanvas(panel_w, panel_h, bg=(20, 20, 20), bg_alpha=0.78, radius=6)
    y = pad
    for line in lines:
        ov.text((pad, y), line, size=size, color=(235, 235, 235), bold=True)
        y += line_h + gap

    x0 = 0 if "left" in corner else width - panel_w
    y0 = 0 if "top" in corner else height - panel_h
    output = frame.copy()
    alpha_composite(output, ov.render(), x0, y0)
    return output


def alpha_composite(frame: np.ndarray, canvas_bgra: np.ndarray, x: int, y: int) -> None:
    """Alpha-blend a BGRA ``canvas`` onto ``frame`` in place at top-left (x, y).

    Clips to the frame bounds, so callers do not have to. No-op if fully off-frame.
    """
    fh, fw = frame.shape[:2]
    ch, cw = canvas_bgra.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + cw), min(fh, y + ch)
    if x1 <= x0 or y1 <= y0:
        return
    sub = canvas_bgra[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    a = sub[:, :, 3:4].astype(np.float32) / 255.0
    frame[y0:y1, x0:x1] = (a * sub[:, :, :3] + (1.0 - a) * roi).astype(np.uint8)
