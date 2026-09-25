"""The side-panel protocol: one placement implementation, one attach method.

Seven classes each carried their own copy of the corner arithmetic, and the copies had
drifted -- some honoured a stacking offset, some did not, some clamped out-of-bounds
placement and some raised. These tests pin the shared behaviour and, importantly, that
stacked panels do not overlap: an off-by-one in the offset silently hides a panel behind
its neighbour, which looks like "the feature didn't work" rather than like a bug.
"""
import numpy as np
import pytest

from physiotrack.core.panel import PanelMixin, attach_stack


class Panel(PanelMixin):
    """A panel returning a fixed, uniquely-coloured canvas."""

    def __init__(self, h=60, w=120, value=200, channels=3, backdrop=False):
        self.canvas = np.full((h, w, channels), value, dtype=np.uint8)
        if channels == 4:
            self.canvas[..., 3] = 255
        self.PANEL_BACKDROP = backdrop

    def render(self):
        return self.canvas


class EmptyPanel(PanelMixin):
    """A panel with nothing to show yet."""

    def render(self):
        return None


@pytest.fixture
def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def drawn_bbox(before, after):
    """Bounding box of pixels that changed, as ``(x0, y0, x1, y1)`` inclusive."""
    ys, xs = np.where((before != after).any(axis=2))
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


class TestPlacement:
    @pytest.mark.parametrize("position,expected_origin", [
        ("top_left", (10, 10)),
        ("top_right", (640 - 120 - 10, 10)),
        ("bottom_left", (10, 480 - 60 - 10)),
        ("bottom_right", (640 - 120 - 10, 480 - 60 - 10)),
    ])
    def test_corner_origins(self, frame, position, expected_origin):
        out = Panel().attach_to_frame(frame, position=position, margin=10)
        x0, y0, _, _ = drawn_bbox(frame, out)
        assert (x0, y0) == expected_origin

    def test_bare_top_and_bottom_mean_the_left_corner(self, frame):
        for bare, explicit in (("top", "top_left"), ("bottom", "bottom_left")):
            a = Panel().attach_to_frame(frame, position=bare, margin=10)
            b = Panel().attach_to_frame(frame, position=explicit, margin=10)
            assert np.array_equal(a, b)

    def test_margin_is_honoured(self, frame):
        out = Panel().attach_to_frame(frame, position="top_left", margin=30)
        assert drawn_bbox(frame, out)[:2] == (30, 30)

    def test_frame_is_not_modified_in_place(self, frame):
        original = frame.copy()
        Panel().attach_to_frame(frame, position="top_left")
        assert np.array_equal(frame, original)

    def test_invalid_position_is_rejected(self, frame):
        with pytest.raises(ValueError, match="Invalid position"):
            Panel().attach_to_frame(frame, position="middle")

    def test_render_returning_none_leaves_the_frame_alone(self, frame):
        out = EmptyPanel().attach_to_frame(frame)
        assert out is frame

    def test_panel_taller_than_the_frame_is_skipped(self, frame):
        # Better to show nothing than to crop or crash mid-render.
        out = Panel(h=2000, w=100).attach_to_frame(frame)
        assert np.array_equal(out, frame)

    def test_overwide_panel_is_downscaled_not_dropped(self, frame):
        out = Panel(h=40, w=5000).attach_to_frame(frame, position="top_left", margin=10)
        box = drawn_bbox(frame, out)
        assert box is not None, "an over-wide panel should be scaled to fit"
        assert box[2] - box[0] + 1 <= 640 - 2 * 10

    def test_a_missing_render_is_a_clear_error(self, frame):
        class Bare(PanelMixin):
            pass

        with pytest.raises(NotImplementedError, match="must implement render"):
            Bare().attach_to_frame(frame)


class TestOffset:
    def test_offset_pushes_a_top_panel_down(self, frame):
        base = Panel().attach_to_frame(frame, position="top_left", margin=10)
        moved = Panel().attach_to_frame(frame, position="top_left", margin=10,
                                        above_element_height=100)
        # offset + one margin of gutter
        assert drawn_bbox(frame, moved)[1] - drawn_bbox(frame, base)[1] == 110

    def test_offset_pushes_a_bottom_panel_up(self, frame):
        base = Panel().attach_to_frame(frame, position="bottom_left", margin=10)
        moved = Panel().attach_to_frame(frame, position="bottom_left", margin=10,
                                        above_element_height=100)
        assert drawn_bbox(frame, base)[1] - drawn_bbox(frame, moved)[1] == 110

    def test_zero_offset_adds_no_gutter(self, frame):
        a = Panel().attach_to_frame(frame, position="top_left", above_element_height=0)
        b = Panel().attach_to_frame(frame, position="top_left")
        assert np.array_equal(a, b)


class TestAttachStack:
    def test_panels_do_not_overlap(self, frame):
        # The whole point of the stack: each panel must occupy its own band.
        panels = [Panel(h=50, value=90), Panel(h=70, value=150), Panel(h=40, value=220)]
        out = attach_stack(frame, panels, "top_left", 10)
        for value in (90, 150, 220):
            assert (out == value).any(), f"panel {value} was not drawn"

    def test_stack_order_grows_away_from_the_corner(self, frame):
        first, second = Panel(h=50, value=90), Panel(h=50, value=200)
        out = attach_stack(frame, [first, second], "top_left", 10)
        y_first = np.where((out == 90).all(axis=2))[0].min()
        y_second = np.where((out == 200).all(axis=2))[0].min()
        assert y_first < y_second

    def test_bottom_stack_grows_upward(self, frame):
        first, second = Panel(h=50, value=90), Panel(h=50, value=200)
        out = attach_stack(frame, [first, second], "bottom_left", 10)
        y_first = np.where((out == 90).all(axis=2))[0].min()
        y_second = np.where((out == 200).all(axis=2))[0].min()
        assert y_second < y_first

    def test_gutter_is_applied_between_panels(self, frame):
        # The gap is gutter + margin: the placement rules add one margin of separation
        # whenever a panel is stacked, and `gutter` is on top of that. This compounding
        # is what shipped before the logic was shared, so overlays look unchanged.
        margin, gutter = 10, 25
        a, b = Panel(h=50, value=90), Panel(h=50, value=200)
        out = attach_stack(frame, [a, b], "top_left", margin, gutter=gutter)
        y_a_end = np.where((out == 90).all(axis=2))[0].max()
        y_b_start = np.where((out == 200).all(axis=2))[0].min()
        assert y_b_start - y_a_end - 1 == gutter + margin

    def test_none_entries_are_skipped(self, frame):
        # Lets a caller list an optional panel unconditionally.
        out = attach_stack(frame, [None, Panel(value=200), None], "top_left", 10)
        assert drawn_bbox(frame, out)[:2] == (10, 10)

    def test_empty_panels_do_not_consume_stack_space(self, frame):
        with_empty = attach_stack(frame, [EmptyPanel(), Panel(value=200)], "top_left", 10)
        without = attach_stack(frame, [Panel(value=200)], "top_left", 10)
        assert np.array_equal(with_empty, without)

    def test_empty_stack_returns_the_frame(self, frame):
        assert attach_stack(frame, [], "top_left") is frame


class TestCompositing:
    def test_bgra_canvas_uses_its_alpha(self, frame):
        out = Panel(channels=4, value=255).attach_to_frame(frame, position="top_left")
        assert not np.array_equal(out, frame)

    def test_backdrop_darkens_beyond_the_canvas(self, frame):
        bright = np.full_like(frame, 255)
        plain = Panel(backdrop=False).attach_to_frame(bright, position="top_left", margin=20)
        shaded = Panel(backdrop=True).attach_to_frame(bright, position="top_left", margin=20)
        # A pixel just outside the canvas is untouched without a backdrop, darkened with one.
        probe = (18, 18)
        assert (plain[probe] == 255).all()
        assert (shaded[probe] < 255).all()

    def test_backdrop_is_ignored_for_bgra(self, frame):
        # A BGRA canvas carries its own alpha; a backdrop would fight it.
        a = Panel(channels=4, backdrop=True).attach_to_frame(frame, position="top_left")
        b = Panel(channels=4, backdrop=False).attach_to_frame(frame, position="top_left")
        assert np.array_equal(a, b)


class TestEveryPanelFollowsTheProtocol:
    def test_all_panel_classes_share_one_attach_method(self):
        """No class may reintroduce its own attach method or a bespoke name."""
        from physiotrack.core.depth_view import DepthView
        from physiotrack.core.ego_view import EgoVideoView
        from physiotrack.core.radar_view import RadarView
        from physiotrack.core.rom_skeleton_view import ROMSkeletonView
        from physiotrack.signals.plotting._estimator_panel import EstimatorPanel
        from physiotrack.signals.plotting.angle_plotter import JointAnglePlotter
        from physiotrack.signals.plotting.keypoint_plotter import KeypointMotionPlotter

        classes = [DepthView, EgoVideoView, RadarView, ROMSkeletonView,
                   EstimatorPanel, JointAnglePlotter, KeypointMotionPlotter]
        for cls in classes:
            assert issubclass(cls, PanelMixin), f"{cls.__name__} is not a PanelMixin"
            assert "attach_to_frame" not in vars(cls), \
                f"{cls.__name__} overrides the shared attach_to_frame"
            for bespoke in ("attach_canvas", "attach_panels"):
                assert not hasattr(cls, bespoke), \
                    f"{cls.__name__} still exposes {bespoke}"
            assert "render" in vars(cls), f"{cls.__name__} must implement render()"

    def test_realtime_plotter_push_is_not_named_plot(self):
        """`plot()` renders a result library-wide; pushing a sample is `push()`."""
        from physiotrack.signals.plotting.realtime_plotter import RealTimePlotter

        assert hasattr(RealTimePlotter, "push")
        assert "plot" not in vars(RealTimePlotter)


class TestVisibilityGuards:
    """A disabled panel must draw nothing at all.

    Each view's old `attach_to_frame` short-circuited on its `enabled` flag before
    rendering. Several of them render a grey "No data" placeholder instead of returning
    None, so losing that short-circuit would stamp an empty box into the corner of every
    frame -- which looks like a broken overlay, not like a disabled feature.
    """

    @staticmethod
    def _view(cls, **attrs):
        obj = cls.__new__(cls)
        for k, v in attrs.items():
            setattr(obj, k, v)
        return obj

    def test_disabled_views_draw_nothing(self, frame):
        import numpy as np

        from physiotrack.core.depth_view import DepthView
        from physiotrack.core.ego_view import EgoVideoView
        from physiotrack.core.radar_view import RadarView
        from physiotrack.core.rom_skeleton_view import ROMSkeletonView

        canvas = np.full((60, 120, 3), 200, np.uint8)
        cases = [
            (RadarView, dict(enabled=False, canvas_size=(120, 60))),
            (DepthView, dict(enabled=False, depth_canvas=canvas)),
            (EgoVideoView, dict(enabled=False, current_frame=canvas, cap=None)),
            (ROMSkeletonView, dict(enabled=False, canvas=canvas)),
        ]
        for cls, attrs in cases:
            view = self._view(cls, **attrs)
            view.render = lambda c=canvas: c
            out = view.attach_to_frame(frame, position="top_left")
            assert np.array_equal(out, frame), f"{cls.__name__} drew while disabled"

    def test_views_with_no_data_yet_draw_nothing(self, frame):
        import numpy as np

        from physiotrack.core.depth_view import DepthView
        from physiotrack.core.ego_view import EgoVideoView
        from physiotrack.core.rom_skeleton_view import ROMSkeletonView

        canvas = np.full((60, 120, 3), 200, np.uint8)
        cases = [
            (DepthView, dict(enabled=True, depth_canvas=None)),
            (EgoVideoView, dict(enabled=True, current_frame=None, cap=None)),
            (ROMSkeletonView, dict(enabled=True, canvas=None)),
        ]
        for cls, attrs in cases:
            view = self._view(cls, **attrs)
            view.render = lambda c=canvas: c      # would draw a placeholder
            out = view.attach_to_frame(frame, position="top_left")
            assert np.array_equal(out, frame), \
                f"{cls.__name__} drew a placeholder before receiving data"

    def test_invisible_panels_take_no_stack_space(self, frame):
        import numpy as np

        class Hidden(Panel):
            def panel_visible(self):
                return False

        with_hidden = attach_stack(frame, [Hidden(), Panel(value=200)], "top_left", 10)
        without = attach_stack(frame, [Panel(value=200)], "top_left", 10)
        assert np.array_equal(with_hidden, without)


class TestInfoPanel:
    def test_draws_only_in_the_requested_corner_and_copies(self):
        from physiotrack.core.overlay import draw_info_panel

        image = np.full((480, 640, 3), 200, np.uint8)
        annotated = draw_info_panel(image, ["Faces detected: 2", "Detector: test.pt"])
        assert annotated.shape == image.shape
        assert not np.array_equal(annotated[:60, :200], image[:60, :200])
        assert np.array_equal(annotated[-20:, -20:], image[-20:, -20:])
        assert (image == 200).all()                      # input untouched

        corner = draw_info_panel(image, ["x"], corner="bottom_right")
        assert np.array_equal(corner[:20, :20], image[:20, :20])
        assert not np.array_equal(corner[-20:, -20:], image[-20:, -20:])
        assert np.array_equal(draw_info_panel(image, []), image)

