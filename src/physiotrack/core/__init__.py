"""Composited overlay panels and the anti-aliased drawing primitives behind them.

These are the side panels the [`Video`][physiotrack.Video] orchestrator lays around
the annotated frame — the bird's-eye floor map, the depth inset, the synchronised
ego-video, and the ROM skeleton canvas — plus the drawing helpers they are built
from, exported so a custom capture loop can compose the same panels.
"""

from .depth_view import DepthView
from .ego_view import EgoVideoView
from .overlay import OverlayCanvas, alpha_composite, draw_info_panel, draw_label
from .radar_view import RadarView
from .rom_skeleton_view import ROMSkeletonView

__all__ = [
    # panels
    "RadarView",
    "DepthView",
    "EgoVideoView",
    "ROMSkeletonView",
    # drawing primitives
    "OverlayCanvas",
    "draw_label",
    "draw_info_panel",
    "alpha_composite",
]
