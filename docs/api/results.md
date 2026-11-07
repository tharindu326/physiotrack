# Result Objects

The unified, typed object model returned by every predictor. A `Result` is an
iterable of [`Instance`](#instance) objects; each `Instance` carries the fields
relevant to its task (box, keypoints, mask, orientation). Depth returns a
[`DepthResult`](#depthresult); the tracker returns a [`TrackResult`](#trackresult).

Rendering is always a method on the result — `result.plot(...)` — so the same
inference output can be drawn many ways without re-running the model.

3D lifting is sequence-level rather than per-frame, so it returns a
[`Pose3DResult`](#pose3dresult) holding the whole `(N, 17, 3)` sequence.

Video processing preserves this object model rather than degrading to plain
dictionaries: [`Video.run()`][physiotrack.Video.run] returns a
[`VideoResults`](#videoresults) sequence of [`FrameResult`](#frameresult), each of
which still exposes its subjects as `Instance` objects. Every result type carries a
[`ResultMeta`](#resultmeta) and round-trips through `to_dict()` / `from_dict()`.

## Serialized vocabulary

[`Result.to_dict()`][physiotrack.Result.to_dict] and
[`TrackResult.to_dict()`][physiotrack.TrackResult.to_dict] both store subjects under
`"instances"`. A serialized instance uses the same names as the Python object:
`box`, `confidence`, `cls`, `cls_name`, and—when present—`id`, `keypoints`, `mask`,
and the face fields `orientation`, `expression`, `gaze` and `quality`. Older examples
that use top-level `"detections"` / `"tracks"` or rename `orientation` to `"pose"` do
not match the current API.

Dense arrays are written only with `include_arrays=True`: masks, segmentation maps and
478-point `"FACEMESH"` keypoints. Without it, `has_mask` / `has_keypoints` record that
the array exists. A `Video` frame record adds `frame_id`, `timestamp`, `task`, and — when
faces were analysed — `faces`, itself a serialized face `Result`.

## Result

::: physiotrack.Result

## Instance

::: physiotrack.Instance

## Keypoints

::: physiotrack.Keypoints

## Keypoint

::: physiotrack.Keypoint

## DepthResult

::: physiotrack.DepthResult

## TrackResult

::: physiotrack.TrackResult

## Pose3DResult

::: physiotrack.Pose3DResult

## FrameResult

::: physiotrack.FrameResult

## VideoResults

::: physiotrack.VideoResults

## ResultMeta

::: physiotrack.ResultMeta
