"""
Unified result objects for Physiotrack predictors.

Every image-based predictor (Detection, Pose, Segmentation, Face) returns a
``Result`` (or ``list[Result]`` for batches). Depth returns a ``DepthResult`` and
the tracker returns a ``TrackResult``. Each result carries the structured data plus
the source frame, and renders its own overlay via ``.plot()`` so that rendering is a
property of the *result*, not configured on the model.

See ``docs/API_REDESIGN.md`` for the full contract.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional

import numpy as np

try:  # cv2 is a hard dependency of the library, but keep results.py importable without it
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


__all__ = [
    "Keypoint",
    "Keypoints",
    "Instance",
    "ResultMeta",
    "Result",
    "DepthResult",
    "TrackResult",
    "FrameResult",
    "VideoResults",
]


@dataclass(frozen=True)
class ResultMeta:
    """Provenance for a result: where it came from and how it was produced.

    Without this, a saved prediction is un-citable — there is no record of which frame
    it came from, which checkpoint produced it, or what the values are measured in. That
    matters most for the physiological and kinematic outputs, where "angle" and
    "velocity" are ambiguous without units.

    Every field is optional, so a predictor can fill in what it knows.

    Attributes:
        frame_index (int | None): Zero-based frame number within the source.
        timestamp (float | None): Seconds from the start of the source.
        fps (float | None): Frame rate of the source, needed to interpret any
            time-derivative computed from a sequence of these results.
        model (str | None): Checkpoint that produced it, ideally the registry path
            (e.g. ``"Pose.ViTPose.COCO.s_coco"``).
        device (str | None): Device it ran on, e.g. ``"cpu"`` or ``"cuda:0"``.
        speed_ms (dict | None): Per-stage timings in milliseconds.
        units (dict | None): Unit for each measured quantity, e.g.
            ``{"keypoints": "pixels", "angles": "degrees"}``.
    """

    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    fps: Optional[float] = None
    model: Optional[str] = None
    device: Optional[str] = None
    speed_ms: Optional[dict] = None
    units: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the populated fields only.

        Returns:
            dict: The non-``None`` fields, so an unannotated result serializes to ``{}``
                rather than a wall of nulls.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ResultMeta":
        """Rebuild from :meth:`to_dict` output, ignoring unknown keys.

        Args:
            data (dict | None): Serialized metadata, or ``None``.

        Returns:
            ResultMeta: The reconstructed metadata; empty when ``data`` is falsy.
        """
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# COCO-17 skeleton edges (used when drawing body keypoints).
_COCO17_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
]


def _name_maps(architecture: Optional[str]):
    """Lazily fetch keypoint id->name maps (avoids import-time circular imports)."""
    from .pose.config import COCO, COCO_WHOLEBODY
    return COCO_WHOLEBODY if architecture == "WHOLEBODY" else COCO


# --------------------------------------------------------------------------- #
# Keypoints
# --------------------------------------------------------------------------- #
class Keypoint:
    """A single body/face/hand landmark with pixel coordinates and confidence.

    A keypoint carries its integer id (indexing into the active skeleton, e.g.
    COCO-17 or COCO-WholeBody-133), its human-readable ``name``, its pixel
    location ``(x, y)`` in the source frame, an optional depth/root-relative
    ``z``, and a detection ``confidence``. Keypoints are produced by pose models
    and grouped inside a [`Keypoints`][physiotrack.Keypoints] collection on each
    [`Instance`][physiotrack.Instance].

    Attributes:
        id (int): Keypoint index in the active skeleton (e.g. ``0`` is
            ``"nose"`` for COCO).
        name (str): Human-readable joint name (e.g. ``"left_shoulder"``), or
            ``"unknown_<id>"`` if the id is not in the name map.
        x (float): X pixel coordinate in the source frame.
        y (float): Y pixel coordinate in the source frame.
        z (float | None): Depth or root-relative Z value when a 3D/depth-aware
            model produced it, otherwise ``None``.
        confidence (float): Detection confidence in ``[0.0, 1.0]``.

    Example:
        ```python
        import physiotrack as pt
        pose = pt.Pose.Person()
        result = pose.predict(frame)
        for kp in result.keypoints[0]:
            print(kp.name, kp.x, kp.y, kp.confidence)
        ```

    See Also:
        [`Keypoints`][physiotrack.Keypoints]: the ordered collection wrapper.
    """

    __slots__ = ("id", "name", "x", "y", "z", "confidence")

    def __init__(self, id: int, name: str, x: float, y: float,
                 confidence: float, z: Optional[float] = None):
        """Construct a keypoint.

        Args:
            id (int): Keypoint index in the active skeleton.
            name (str): Human-readable joint name.
            x (float): X pixel coordinate in the source frame.
            y (float): Y pixel coordinate in the source frame.
            confidence (float): Detection confidence in ``[0.0, 1.0]``.
            z (float, optional): Depth or root-relative Z value. Defaults to
                ``None`` (2D-only keypoint).
        """
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.confidence = confidence

    def __repr__(self) -> str:
        z = "" if self.z is None else f", z={self.z:.1f}"
        return (f"Keypoint(id={self.id}, name='{self.name}', "
                f"x={self.x:.1f}, y={self.y:.1f}{z}, conf={self.confidence:.3f})")


class Keypoints:
    """Ordered collection of [`Keypoint`][physiotrack.Keypoint] for one subject.

    Wraps the per-instance landmarks and supports three lookup styles: positional
    (``kps[0]``), by skeleton id (``kps.by_id(5)``), and by name
    (``kps.by_name("left_shoulder")``). It also exposes vectorized NumPy views
    (:attr:`xy`, :attr:`xyz`, :attr:`conf`) for numeric work. Iteration and
    ``len()`` follow insertion order, which matches the model's skeleton order.

    Attributes:
        architecture (str): Skeleton the ids/names come from. ``"WHOLEBODY"``
            uses the COCO-WholeBody-133 name map; anything else uses COCO-17.

    Example:
        ```python
        import physiotrack as pt
        pose = pt.Pose.Person()
        kps = pose.predict(frame).keypoints[0]
        nose = kps.by_name("nose")
        print(len(kps), kps.xy.shape)        # e.g. 133 (133, 2)
        ```

    See Also:
        [`Keypoint`][physiotrack.Keypoint]: the individual landmark.
        [`Instance`][physiotrack.Instance]: holds one ``Keypoints`` as ``.keypoints``.
    """

    def __init__(self, keypoints_data: List[dict], architecture: str = "WHOLEBODY"):
        """Build a keypoint collection from raw model output.

        Args:
            keypoints_data (list[dict]): One dict per keypoint with keys
                ``"id"``, ``"x"``, ``"y"``, ``"confidence"``, and optionally
                ``"z"``. Ids are mapped to names via the architecture's name map.
            architecture (str, optional): Skeleton naming to apply. ``"WHOLEBODY"``
                uses COCO-WholeBody-133 names; any other value uses COCO-17.
                Defaults to ``"WHOLEBODY"``.
        """
        self.architecture = architecture
        names = _name_maps(architecture)
        self._ordered: List[Keypoint] = []
        self._by_id: Dict[int, Keypoint] = {}
        self._by_name: Dict[str, Keypoint] = {}

        for kp in keypoints_data:
            name = names.get(str(kp["id"]), f"unknown_{kp['id']}")
            keypoint = Keypoint(
                id=kp["id"], name=name, x=kp["x"], y=kp["y"],
                confidence=kp["confidence"], z=kp.get("z"),
            )
            self._ordered.append(keypoint)
            self._by_id[keypoint.id] = keypoint
            self._by_name[name] = keypoint

    # -- access -------------------------------------------------------------- #
    def by_id(self, keypoint_id: int) -> Optional[Keypoint]:
        """Look up a keypoint by its skeleton id.

        Args:
            keypoint_id (int): Skeleton index to fetch (e.g. ``0`` for nose).

        Returns:
            Keypoint | None: The matching [`Keypoint`][physiotrack.Keypoint], or
                ``None`` if no keypoint with that id is present.
        """
        return self._by_id.get(keypoint_id)

    def by_name(self, keypoint_name: str) -> Optional[Keypoint]:
        """Look up a keypoint by its joint name.

        Args:
            keypoint_name (str): Joint name to fetch (e.g. ``"left_shoulder"``).

        Returns:
            Keypoint | None: The matching [`Keypoint`][physiotrack.Keypoint], or
                ``None`` if no keypoint with that name is present.
        """
        return self._by_name.get(keypoint_name)

    def __getitem__(self, index: int) -> Keypoint:
        """Return the keypoint at the given positional index (skeleton order).

        Args:
            index (int): Zero-based position in skeleton order.

        Returns:
            Keypoint: The keypoint at ``index``.
        """
        return self._ordered[index]

    def __iter__(self):
        """Iterate over keypoints in skeleton order.

        Yields:
            Keypoint: Each [`Keypoint`][physiotrack.Keypoint] in order.
        """
        return iter(self._ordered)

    def __len__(self) -> int:
        """Return the number of keypoints in the collection.

        Returns:
            int: Count of keypoints (e.g. ``17`` for COCO, ``133`` for WholeBody).
        """
        return len(self._ordered)

    # -- vectorized views ---------------------------------------------------- #
    @property
    def xy(self) -> np.ndarray:
        """Pixel coordinates of every keypoint, in skeleton order.

        Returns:
            np.ndarray: Float32 array of shape ``(N, 2)`` holding ``(x, y)`` pixel
                coordinates for the ``N`` keypoints.
        """
        return np.array([[k.x, k.y] for k in self._ordered], dtype=np.float32)

    @property
    def xyz(self) -> Optional[np.ndarray]:
        """3D coordinates of every keypoint, when depth/Z is available.

        Returns:
            np.ndarray | None: Float32 array of shape ``(N, 3)`` holding
                ``(x, y, z)`` for the ``N`` keypoints, or ``None`` if the
                keypoints are 2D-only (first keypoint has no ``z``).
        """
        if not self._ordered or self._ordered[0].z is None:
            return None
        return np.array([[k.x, k.y, k.z] for k in self._ordered], dtype=np.float32)

    @property
    def conf(self) -> np.ndarray:
        """Confidence of every keypoint, in skeleton order.

        Returns:
            np.ndarray: Float32 array of shape ``(N,)`` with per-keypoint
                confidences in ``[0.0, 1.0]``.
        """
        return np.array([k.confidence for k in self._ordered], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Instance (one detected subject)
# --------------------------------------------------------------------------- #
class Instance:
    """A single detected subject within a frame.

    ``Instance`` is the per-subject record inside a [`Result`][physiotrack.Result].
    Which fields are populated depends on the task: detection sets
    ``box``/``confidence``/``cls``/``cls_name``; pose adds ``keypoints``;
    segmentation may add a per-instance ``mask``; face orientation adds
    ``orientation``; tracking sets a persistent ``id``. The extended face-analysis
    pipeline stores its per-subject descriptors in ``face_features``. Unused fields
    are ``None``.

    Attributes:
        id (int | None): Persistent track id (set by the tracker), otherwise
            ``None``.
        box (np.ndarray | None): Bounding box ``[x1, y1, x2, y2]`` in pixels,
            shape ``(4,)``, or ``None``.
        confidence (float | None): Detection confidence in ``[0.0, 1.0]``, or
            ``None``.
        cls (int | None): Integer class id, or ``None``.
        cls_name (str | None): Human-readable class name (e.g. ``"person"``), or
            ``None``.
        keypoints (Keypoints | None): Pose landmarks for this subject as a
            [`Keypoints`][physiotrack.Keypoints], or ``None``.
        mask (np.ndarray | None): Binary instance mask of shape ``(H, W)``, or
            ``None``.
        orientation (dict | None): Head pose dict ``{"yaw", "pitch", "roll"}``
            in degrees, or ``None``.
        face_features (dict | None): Structured face-analysis descriptors associated
            with this subject, such as quality, eye/blink, gaze, mouth, emotion,
            and semantic-region information, or ``None`` when unavailable.

    Example:
        ```python
        import physiotrack as pt
        result = pt.Pose.Person().predict(frame)
        inst = result[0]
        print(inst.box, inst.confidence)
        if inst.keypoints is not None:
            print(inst.keypoints.by_name("nose"))
        ```

    See Also:
        [`Result`][physiotrack.Result]: the frame-level container of instances.
        [`Keypoints`][physiotrack.Keypoints]: the ``keypoints`` field type.
    """

    __slots__ = ("id", "box", "confidence", "cls", "cls_name",
                 "keypoints", "mask", "orientation", "face_features")

    def __init__(self, *, id: Optional[int] = None,
                 box: Optional[np.ndarray] = None,
                 confidence: Optional[float] = None,
                 cls: Optional[int] = None,
                 cls_name: Optional[str] = None,
                 keypoints: Optional[Keypoints] = None,
                 mask: Optional[np.ndarray] = None,
                 orientation: Optional[dict] = None,
                 face_features: Optional[Dict[str, Any]] = None):
        """Construct an instance (all fields keyword-only and optional).

        Args:
            id (int, optional): Persistent track id. Defaults to ``None``.
            box (np.ndarray, optional): Bounding box ``[x1, y1, x2, y2]`` of
                shape ``(4,)``. Defaults to ``None``.
            confidence (float, optional): Detection confidence in ``[0.0, 1.0]``.
                Defaults to ``None``.
            cls (int, optional): Integer class id. Defaults to ``None``.
            cls_name (str, optional): Human-readable class name. Defaults to
                ``None``.
            keypoints (Keypoints, optional): Pose landmarks. Defaults to ``None``.
            mask (np.ndarray, optional): Binary instance mask of shape ``(H, W)``.
                Defaults to ``None``.
            orientation (dict, optional): Head pose ``{"yaw", "pitch", "roll"}``
                in degrees. Defaults to ``None``.
            face_features (dict, optional): Structured face-analysis descriptors,
                such as quality, eye/blink, gaze, mouth, emotion, and semantic-region
                information. Defaults to ``None``.
        """
        self.id = id
        self.box = box
        self.confidence = confidence
        self.cls = cls
        self.cls_name = cls_name
        self.keypoints = keypoints
        self.mask = mask
        self.orientation = orientation
        self.face_features = face_features

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Serialize the instance to a JSON-friendly dict.

        Keys match the attribute names, so the serialized form and the object model use
        one vocabulary: ``box`` (not ``bbox``) and ``orientation`` (not ``pose``). Only
        populated fields are emitted.

        Args:
            include_arrays (bool, optional): Include :attr:`mask` as a nested list.
                Defaults to ``False``, since a per-instance mask is megabytes of JSON.

        Returns:
            dict: Any of ``id``, ``box`` (``[x1, y1, x2, y2]``), ``confidence``, ``cls``,
                ``cls_name``, ``keypoints`` (list of ``{"id", "x", "y", "confidence",
                "z"?}``), ``orientation`` (``{"yaw", "pitch", "roll"}``),
                ``face_features`` (structured face-analysis descriptors), and ``mask``
                when requested. ``has_mask`` is always present when a mask exists, so a
                consumer can tell an omitted mask from an absent one.
        """
        out: Dict[str, Any] = {}
        if self.id is not None:
            out["id"] = self.id
        if self.box is not None:
            out["box"] = np.asarray(self.box).tolist()
        if self.confidence is not None:
            out["confidence"] = float(self.confidence)
        if self.cls is not None:
            out["cls"] = int(self.cls)
        if self.cls_name is not None:
            out["cls_name"] = self.cls_name
        if self.keypoints is not None:
            out["keypoints"] = [
                {"id": k.id, "x": k.x, "y": k.y, "confidence": k.confidence,
                 **({"z": k.z} if k.z is not None else {})}
                for k in self.keypoints
            ]
        if self.orientation is not None:
            out["orientation"] = self.orientation
        if self.face_features is not None:
            out["face_features"] = self.face_features
        if self.mask is not None:
            out["has_mask"] = True
            if include_arrays:
                out["mask"] = np.asarray(self.mask).tolist()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any], architecture: str = "WHOLEBODY") -> "Instance":
        """Rebuild an instance from :meth:`to_dict` output.

        Args:
            data (dict): A serialized instance.
            architecture (str, optional): Skeleton naming for the keypoints, so their
                ``name`` fields are restored. Defaults to ``"WHOLEBODY"``.

        Returns:
            Instance: The reconstructed instance. A mask is restored only if the dict
                carries the array (see ``include_arrays``).
        """
        box = data.get("box")
        mask = data.get("mask")
        keypoints = data.get("keypoints")
        return cls(
            id=data.get("id"),
            box=(np.asarray(box, dtype=np.float32) if box is not None else None),
            confidence=data.get("confidence"),
            cls=data.get("cls"),
            cls_name=data.get("cls_name"),
            keypoints=(Keypoints(keypoints, architecture) if keypoints else None),
            mask=(np.asarray(mask) if mask is not None else None),
            orientation=data.get("orientation"),
            face_features=data.get("face_features"),
        )

    def __repr__(self) -> str:
        parts = [f"id={self.id}"]
        if self.box is not None:
            parts.append(f"box={np.round(np.asarray(self.box), 1).tolist()}")
        if self.confidence is not None:
            parts.append(f"conf={self.confidence:.3f}")
        if self.keypoints is not None:
            parts.append(f"keypoints={len(self.keypoints)}")
        if self.mask is not None:
            parts.append("mask=yes")
        if self.orientation is not None:
            parts.append(f"orientation={self.orientation}")
        if self.face_features is not None:
            parts.append(f"face_features={list(self.face_features.keys())}")
        return f"Instance({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# Result (detect / pose / segment / face)
# --------------------------------------------------------------------------- #
class Result:
    """Per-frame result shared by all image tasks (detect, pose, segment, face).

    Returned by every image-based predictor for a single frame (a
    ``list[Result]`` is returned for a batch). A ``Result`` bundles the source
    frame with the detected [`Instance`][physiotrack.Instance] objects and,
    for segmentation, a class-index map. It behaves like a sequence of instances
    (``len(result)``, ``result[i]``, iteration), exposes convenience views
    (:attr:`boxes`, :attr:`keypoints`), serializes via :meth:`to_dict`, and
    renders its own overlay via :meth:`plot` — so rendering is a property of the
    result, not configured on the model.

    Attributes:
        orig_img (np.ndarray): Source BGR frame ``(H, W, 3)`` the result was
            computed from.
        instances (list[Instance]): Detected subjects in the frame.
        task (str): Task that produced this result — one of ``"detect"``,
            ``"pose"``, ``"segment"``, ``"face"``.
        architecture (str | None): Model/skeleton hint (e.g. ``"WHOLEBODY"``) used
            when interpreting keypoints, or ``None``.
        seg_map (np.ndarray | None): Class-index map of shape ``(H, W)`` for
            segmentation tasks, otherwise ``None``.
        names (dict[int, str] | None): Class-id to class-name mapping, or ``None``.
        palette (np.ndarray | None): Optional ``(K, 3)`` RGB palette used to
            colorize ``seg_map`` (e.g. face parsing). When ``None`` the default
            segmentation palette is used.

    Example:
        ```python
        import physiotrack as pt
        det = pt.Detection.Person()
        result = det.predict(frame)
        print(len(result), result.boxes.shape)   # e.g. 3 (3, 4)
        annotated = result.plot(conf=True)
        data = result.to_dict()
        ```

    See Also:
        [`Instance`][physiotrack.Instance]: a single subject in the result.
        [`DepthResult`][physiotrack.DepthResult]: depth-task counterpart.
        [`TrackResult`][physiotrack.TrackResult]: tracker counterpart.
    """

    def __init__(self, *, orig_img: np.ndarray, instances: List[Instance],
                 task: str, architecture: Optional[str] = None,
                 seg_map: Optional[np.ndarray] = None,
                 names: Optional[Dict[int, str]] = None,
                 palette: Optional[np.ndarray] = None,
                 meta: Optional["ResultMeta"] = None):
        """Construct a per-frame result (all fields keyword-only).

        Args:
            orig_img (np.ndarray): Source BGR frame of shape ``(H, W, 3)``.
            instances (list[Instance]): Detected subjects for this frame.
            task (str): Producing task: ``"detect"``, ``"pose"``, ``"segment"``,
                or ``"face"``.
            architecture (str, optional): Skeleton/model hint (e.g.
                ``"WHOLEBODY"``). Defaults to ``None``.
            seg_map (np.ndarray, optional): Class-index map ``(H, W)`` for
                segmentation. Defaults to ``None``.
            names (dict[int, str], optional): Class-id to name map. Defaults to
                ``None``.
            palette (np.ndarray, optional): ``(K, 3)`` RGB palette for colorizing
                ``seg_map``. Defaults to ``None`` (default palette).
            meta (ResultMeta, optional): Provenance — where in the video this came
                from, which model produced it, on what device, and how long it took.
                Defaults to an empty [`ResultMeta`][physiotrack.ResultMeta].
        """
        self.orig_img = orig_img
        self.instances = instances
        self.task = task
        self.architecture = architecture
        self.seg_map = seg_map
        self.names = names
        # Optional (K, 3) RGB palette for colorizing ``seg_map`` (e.g. face parsing).
        # When None, the default segmentation palette is used.
        self.palette = palette
        self.meta = meta if meta is not None else ResultMeta()

    # -- container protocol -------------------------------------------------- #
    def __iter__(self):
        """Iterate over the detected instances in the frame.

        Yields:
            Instance: Each [`Instance`][physiotrack.Instance] in ``instances``.
        """
        return iter(self.instances)

    def __len__(self) -> int:
        """Return the number of detected instances.

        Returns:
            int: Count of instances in the frame.
        """
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        """Return the instance at the given index.

        Args:
            index (int): Zero-based instance index.

        Returns:
            Instance: The [`Instance`][physiotrack.Instance] at ``index``.
        """
        return self.instances[index]

    def __repr__(self) -> str:
        return (f"Result(task='{self.task}', instances={len(self.instances)}"
                f"{', architecture=' + repr(self.architecture) if self.architecture else ''})")

    # -- convenience views --------------------------------------------------- #
    @property
    def boxes(self) -> np.ndarray:
        """Bounding boxes of all instances that have one.

        Returns:
            np.ndarray: Float32 array of shape ``(M, 4)`` with rows
                ``[x1, y1, x2, y2]`` for the ``M`` instances that have a box; an
                empty ``(0, 4)`` array when there are none.
        """
        boxes = [i.box for i in self.instances if i.box is not None]
        return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), np.float32)

    @property
    def keypoints(self) -> List[Keypoints]:
        """Keypoint collections for all instances that have pose landmarks.

        Returns:
            list[Keypoints]: One [`Keypoints`][physiotrack.Keypoints] per instance
                that has keypoints (instances without pose data are skipped).
        """
        return [i.keypoints for i in self.instances if i.keypoints is not None]

    # -- serialization ------------------------------------------------------- #
    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Serialize the result to a plain, JSON-friendly dict.

        The subjects are under ``"instances"`` — the same name as the
        :attr:`instances` attribute — and each one uses the attribute names rather than
        aliases, so the object model and the serialized form share one vocabulary. Pair
        with [`from_dict`][physiotrack.Result.from_dict] to round-trip.

        Args:
            include_arrays (bool, optional): Include the large arrays: per-instance
                masks and the frame-level ``seg_map``. Defaults to ``False``, which
                keeps the output small enough for JSON while still recording that a
                mask or map exists.

        Returns:
            dict: ``"task"`` (str) and ``"instances"`` (list of
                [`Instance.to_dict`][physiotrack.Instance.to_dict] outputs), plus
                ``"architecture"``, ``"names"``, ``"has_seg_map"`` and ``"seg_map"``
                when applicable.

        Example:
            ```python
            import physiotrack as pt
            data = pt.Pose.Person().predict(frame).to_dict()
            data["task"], len(data["instances"])

            restored = pt.Result.from_dict(data)
            ```
        """
        out: Dict[str, Any] = {
            "task": self.task,
            "instances": [i.to_dict(include_arrays=include_arrays) for i in self.instances],
        }
        if self.architecture is not None:
            out["architecture"] = self.architecture
        if self.names is not None:
            out["names"] = self.names
        if self.seg_map is not None:
            out["has_seg_map"] = True
            if include_arrays:
                out["seg_map"] = np.asarray(self.seg_map).tolist()
        meta = self.meta.to_dict()
        if meta:
            out["meta"] = meta
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  orig_img: Optional[np.ndarray] = None) -> "Result":
        """Rebuild a result from :meth:`to_dict` output.

        Lets serialized predictions be reloaded into the object model — so a JSON file
        written by an earlier run can be re-analysed, plotted, or fed to the signals
        functions without hand-parsing it.

        Args:
            data (dict): A serialized result.
            orig_img (np.ndarray, optional): Source frame to attach, since the pixels are
                not serialized. Required for [`plot`][physiotrack.Result.plot]; defaults
                to ``None``.

        Returns:
            Result: The reconstructed result. Arrays absent from ``data`` (a mask or
                ``seg_map`` omitted by ``include_arrays=False``) stay ``None``.

        Raises:
            KeyError: If ``data`` has no ``"task"`` key.
        """
        if "task" not in data:
            raise KeyError(
                "Serialized result is missing the 'task' key; expected the output of "
                "Result.to_dict()."
            )
        architecture = data.get("architecture", "WHOLEBODY")
        seg_map = data.get("seg_map")
        return cls(
            orig_img=orig_img,
            instances=[Instance.from_dict(d, architecture)
                       for d in data.get("instances", [])],
            task=data["task"],
            architecture=data.get("architecture"),
            seg_map=(np.asarray(seg_map) if seg_map is not None else None),
            names=data.get("names"),
            meta=ResultMeta.from_dict(data.get("meta")),
        )

    def to_json(self, path=None, *, include_arrays: bool = False, indent: int = 2):
        """Serialize to JSON, optionally writing it to a file.

        Args:
            path (str | os.PathLike, optional): Destination file. Defaults to ``None``,
                which returns the JSON string instead of writing.
            include_arrays (bool, optional): Include masks and ``seg_map``. Defaults to
                ``False``.
            indent (int, optional): JSON indentation. Defaults to ``2``.

        Returns:
            str | None: The JSON string when ``path`` is ``None``, otherwise ``None``.
        """
        import json

        payload = json.dumps(self.to_dict(include_arrays=include_arrays), indent=indent)
        if path is None:
            return payload
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return None

    def save(self, path, **plot_kwargs) -> None:
        """Render the overlay and write it to an image file.

        Args:
            path (str | os.PathLike): Destination image path; the extension selects the
                format.
            **plot_kwargs (Any): Forwarded to [`plot`][physiotrack.Result.plot].

        Raises:
            RuntimeError: If OpenCV is unavailable, or the file could not be written.
        """
        if cv2 is None:  # pragma: no cover - cv2 is a hard dependency
            raise RuntimeError("Saving an overlay requires OpenCV (cv2).")
        if not cv2.imwrite(str(path), self.plot(**plot_kwargs)):
            raise RuntimeError(f"Could not write the annotated image to {path!r}.")

    # -- rendering ----------------------------------------------------------- #
    def plot(self, *, boxes: bool = True, labels: bool = True,
             keypoints: bool = True, masks: bool = True, conf: bool = False,
             color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Render an annotated copy of the source frame.

        Drawing is controlled here rather than on the model, so the same result can
        be drawn in different ways without re-running inference. Boxes, class/id
        labels, pose skeletons, segmentation masks, and head-pose axes are drawn
        based on what each instance carries and the toggles below. The original
        frame is not modified.

        Args:
            boxes (bool, optional): Draw bounding boxes. Defaults to ``True``.
            labels (bool, optional): Draw class/track-id labels above boxes (only
                when ``boxes`` is also drawn). Defaults to ``True``.
            keypoints (bool, optional): Draw pose keypoints and the COCO-17
                skeleton. Defaults to ``True``.
            masks (bool, optional): Blend segmentation masks (only for the
                ``"segment"`` task). Defaults to ``True``.
            conf (bool, optional): Append the detection confidence to labels.
                Defaults to ``False``.
            color (tuple, optional): Box/label BGR color. Defaults to
                ``(0, 255, 0)`` (green).
            thickness (int, optional): Box line thickness in pixels. Defaults to
                ``2``.

        Returns:
            np.ndarray: A new annotated BGR image of shape ``(H, W, 3)``.

        Raises:
            RuntimeError: If OpenCV (``cv2``) is not installed.

        Example:
            ```python
            import physiotrack as pt
            result = pt.Detection.Person().predict(frame)
            annotated = result.plot(conf=True, color=(0, 0, 255))
            ```

        Note:
            Head-pose axes are always drawn for instances that carry an
            ``orientation``, regardless of the toggles above.
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for Result.plot().")
        img = self.orig_img.copy()

        if masks and self.task == "segment":
            img = self._draw_masks(img)

        for inst in self.instances:
            if boxes and inst.box is not None:
                x1, y1, x2, y2 = [int(v) for v in inst.box]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                if labels:
                    self._draw_label(img, inst, x1, y1, conf, color)
            if keypoints and inst.keypoints is not None:
                self._draw_keypoints(img, inst.keypoints)
            if inst.orientation is not None:
                self._draw_orientation(img, inst)

        return img

    # -- drawing helpers ----------------------------------------------------- #
    @staticmethod
    def _draw_label(img, inst, x1, y1, show_conf, color):
        label = inst.cls_name if inst.cls_name else (
            f"id {inst.id}" if inst.id is not None else inst.task if False else "")
        if inst.id is not None and inst.cls_name:
            label = f"{inst.cls_name} {inst.id}"
        if show_conf and inst.confidence is not None:
            label = f"{label} {inst.confidence:.2f}".strip()
        if not label:
            return
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, label, (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_keypoints(img, keypoints: "Keypoints", conf_thresh: float = 0.3):
        # skeleton (body-17 only; higher ids drawn as points)
        for a, b in _COCO17_SKELETON:
            ka, kb = keypoints.by_id(a), keypoints.by_id(b)
            if ka and kb and ka.confidence > conf_thresh and kb.confidence > conf_thresh:
                cv2.line(img, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)),
                         (255, 128, 0), 2, cv2.LINE_AA)
        for kp in keypoints:
            if kp.confidence > conf_thresh:
                cv2.circle(img, (int(kp.x), int(kp.y)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    def _draw_masks(self, img):
        # Class-index map (the common segmentation output): colorize and blend.
        if self.seg_map is not None:
            try:
                if self.palette is not None:
                    # Palette-based colorizing (e.g. SegFace face parsing). Blend only
                    # over foreground (class > 0) so the background frame is untouched.
                    idx = np.clip(self.seg_map, 0, len(self.palette) - 1)
                    color_map = cv2.cvtColor(self.palette[idx].astype(np.uint8),
                                             cv2.COLOR_RGB2BGR)
                    if color_map.shape[:2] != img.shape[:2]:
                        color_map = cv2.resize(color_map, (img.shape[1], img.shape[0]),
                                               interpolation=cv2.INTER_NEAREST)
                    fg = (self.seg_map > 0)
                    if fg.shape[:2] != img.shape[:2]:
                        fg = cv2.resize(fg.astype(np.uint8), (img.shape[1], img.shape[0]),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                    blended = cv2.addWeighted(color_map, 0.5, img, 0.5, 0)
                    img = np.where(fg[..., None], blended, img)
                else:
                    from .modules import draw_segmentation_map
                    color_map = draw_segmentation_map(self.seg_map)
                    if color_map.shape[:2] != img.shape[:2]:
                        color_map = cv2.resize(color_map, (img.shape[1], img.shape[0]))
                    img = cv2.addWeighted(color_map, 0.5, img, 0.5, 0)
            except Exception as exc:
                # Never let an overlay failure abort a caller's render loop, but do
                # not hide it either: a silently mask-less plot is indistinguishable
                # from a backend that produced no segmentation.
                warnings.warn(
                    f"Could not draw the segmentation overlay: {exc!r}. "
                    f"The returned image has no masks drawn.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        # Per-instance binary masks (if a backend provides them).
        rng = np.random.default_rng(0)
        for inst in self.instances:
            if inst.mask is None:
                continue
            mask = inst.mask.astype(bool)
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(inst.mask.astype(np.uint8),
                                  (img.shape[1], img.shape[0])).astype(bool)
            tint = rng.integers(64, 255, size=3).tolist()
            overlay = img.copy()
            overlay[mask] = tint
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        return img

    @staticmethod
    def _draw_orientation(img, inst):
        from .modules._6DRepNet360.utils import draw_axis
        o = inst.orientation or {}
        if not all(k in o for k in ("yaw", "pitch", "roll")):
            return
        if inst.box is not None:
            x1, y1, x2, y2 = [int(v) for v in inst.box]
            tdx, tdy = (x1 + x2) // 2, (y1 + y2) // 2
        else:
            tdx = tdy = None
        draw_axis(img, o["yaw"], o["pitch"], o["roll"], tdx=tdx, tdy=tdy)


# --------------------------------------------------------------------------- #
# DepthResult
# --------------------------------------------------------------------------- #
class DepthResult:
    """Dense depth result: the raw depth map plus colorization via :meth:`plot`.

    Returned by depth predictors for a single frame. Holds the source frame and a
    per-pixel depth map. Use :meth:`normalized` for a ``[0, 1]`` map or
    :meth:`plot` for a colorized BGR image ready to display.

    Attributes:
        orig_img (np.ndarray): Source BGR frame ``(H, W, 3)``.
        depth (np.ndarray): Raw per-pixel depth map of shape ``(H, W)``; larger
            values are nearer or farther depending on the backend.

    Example:
        ```python
        import physiotrack as pt
        d = pt.Depth.DepthAnythingV2Base().predict(frame)
        raw = d.depth                       # (H, W) float depth
        colored = d.plot(colormap="viridis")
        ```

    See Also:
        [`Result`][physiotrack.Result]: image-task counterpart.
    """

    _COLORMAPS = {
        "inferno": "COLORMAP_INFERNO", "viridis": "COLORMAP_VIRIDIS",
        "magma": "COLORMAP_MAGMA", "plasma": "COLORMAP_PLASMA", "jet": "COLORMAP_JET",
    }

    def __init__(self, *, orig_img: np.ndarray, depth: np.ndarray):
        """Construct a depth result (fields keyword-only).

        Args:
            orig_img (np.ndarray): Source BGR frame of shape ``(H, W, 3)``.
            depth (np.ndarray): Raw depth map of shape ``(H, W)``.
        """
        self.orig_img = orig_img
        self.depth = depth

    def __repr__(self) -> str:
        return f"DepthResult(shape={tuple(self.depth.shape)})"

    def normalized(self) -> np.ndarray:
        """Return the depth map min-max normalized to ``[0, 1]``.

        Returns:
            np.ndarray: Float32 map of shape ``(H, W)`` scaled to ``[0.0, 1.0]``.
                Returns an all-zeros map when the depth is (near-)constant.
        """
        d = self.depth.astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < 1e-8:
            return np.zeros_like(d)
        return (d - lo) / (hi - lo)

    def plot(self, *, colormap: str = "inferno") -> np.ndarray:
        """Colorize the depth map into a displayable BGR image.

        The depth is min-max normalized (see :meth:`normalized`) and mapped
        through an OpenCV colormap.

        Args:
            colormap (str, optional): Colormap name. One of ``"inferno"``,
                ``"viridis"``, ``"magma"``, ``"plasma"``, ``"jet"``. Unknown
                names fall back to ``"inferno"``. Defaults to ``"inferno"``.

        Returns:
            np.ndarray: Colorized BGR image of shape ``(H, W, 3)``, dtype uint8.

        Raises:
            RuntimeError: If OpenCV (``cv2``) is not installed.

        Example:
            ```python
            import physiotrack as pt
            colored = pt.Depth.DepthAnythingV2Base().predict(frame).plot(colormap="magma")
            ```
        """
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for DepthResult.plot().")
        norm = (self.normalized() * 255).astype(np.uint8)
        cmap = getattr(cv2, self._COLORMAPS.get(colormap, "COLORMAP_INFERNO"))
        return cv2.applyColorMap(norm, cmap)

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Serialize the depth result.

        Args:
            include_arrays (bool, optional): Include the raw depth map as a nested list.
                Defaults to ``False``, because a full-resolution depth map is tens of
                megabytes of JSON. When omitted, ``shape`` still records what was
                produced, so the result can be round-tripped structurally.

        Returns:
            dict: ``"task"`` (``"depth"``), ``"shape"`` (``[H, W]``), ``"relative"``
                (always ``True`` — see the note), and ``"depth"`` when requested.

        Note:
            The values are **relative** depth, not metres: larger means nearer, and the
            scale is arbitrary and not comparable between frames. This is recorded in the
            output so a downstream consumer cannot mistake it for a metric map.
        """
        out: Dict[str, Any] = {
            "task": "depth",
            "shape": list(self.depth.shape),
            "relative": True,
        }
        if include_arrays:
            out["depth"] = np.asarray(self.depth).tolist()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  orig_img: Optional[np.ndarray] = None) -> "DepthResult":
        """Rebuild a depth result from :meth:`to_dict` output.

        Args:
            data (dict): A serialized depth result.
            orig_img (np.ndarray, optional): Source frame to attach. Defaults to ``None``.

        Returns:
            DepthResult: The reconstructed result.

        Raises:
            KeyError: If ``data`` carries no ``"depth"`` array, which happens when it was
                serialized with ``include_arrays=False``. The map cannot be recovered from
                the shape alone.
        """
        if "depth" not in data:
            raise KeyError(
                "Serialized depth result has no 'depth' array (it was written with "
                "include_arrays=False), so the map cannot be restored. Re-serialize with "
                "include_arrays=True to round-trip."
            )
        return cls(orig_img=orig_img, depth=np.asarray(data["depth"], dtype=float))


# --------------------------------------------------------------------------- #
# Pose3DResult
# --------------------------------------------------------------------------- #
class Pose3DResult:
    """Lifted 3D pose sequence: ``(N, 17, 3)`` joints in Human3.6M order.

    Returned by [`Pose3D.predict`][physiotrack.Pose3D]. Unlike the per-frame result
    objects, this one is inherently **sequence-level**: a temporal lifter needs a window
    of 2D frames to produce each 3D frame, so a single frame cannot be lifted in
    isolation. Indexing and iteration therefore walk *frames*, yielding ``(17, 3)``
    arrays.

    Coordinates are **root-relative** and in an arbitrary scale unless the backend was
    run in pixel mode — they are not metric, and not comparable between videos. This is
    recorded in :meth:`to_dict` so a consumer cannot mistake them for millimetres.

    Attributes:
        poses (np.ndarray): ``(N, 17, 3)`` joint positions in Human3.6M order.
        fps (float | None): Frame rate of the source sequence, when known.
        view (CanonicalView | None): The canonical viewpoint the poses were rotated
            to, or ``None`` if no canonicalization was applied.
        meta (ResultMeta): Model/device/timing metadata.

    Example:
        ```python
        import physiotrack as pt

        lifter = pt.Pose3D(model=pt.Models.Pose3D.MotionBERT.mb_ft_h36m_global_lite)
        res = lifter.predict(keypoints_2d, fps=30)
        res.poses.shape                     # (N, 17, 3)
        res.by_name("left_wrist").shape     # (N, 3) -- one joint over time
        res[0].shape                        # (17, 3) -- one frame
        ```

    See Also:
        [`canonicalize_pose`][physiotrack.canonicalize_pose]: rotate a sequence to a
        fixed viewpoint.
    """

    def __init__(self, *, poses: np.ndarray, fps: Optional[float] = None,
                 view: Any = None, meta: Optional["ResultMeta"] = None):
        """Construct a 3D pose sequence (fields keyword-only).

        Args:
            poses (np.ndarray): ``(N, 17, 3)`` joint positions in Human3.6M order.
            fps (float, optional): Source frame rate. Defaults to ``None``.
            view (CanonicalView, optional): Canonical viewpoint applied. Defaults to
                ``None``.
            meta (ResultMeta, optional): Model/device metadata. Defaults to ``None``.

        Raises:
            ValueError: If ``poses`` is not ``(N, 17, 3)``.
        """
        arr = np.asarray(poses, dtype=float)
        if arr.ndim != 3 or arr.shape[1:] != (17, 3):
            raise ValueError(
                f"3D poses must have shape (N, 17, 3) in Human3.6M joint order, got "
                f"{tuple(arr.shape)}."
            )
        self.poses = arr
        self.fps = float(fps) if fps is not None else None
        self.view = view
        self.meta = meta if meta is not None else ResultMeta(
            units={"poses": "relative"})

    def __repr__(self) -> str:
        view = getattr(self.view, "name", self.view)
        return (f"Pose3DResult(frames={len(self)}, joints=17, fps={self.fps}, "
                f"view={view})")

    def __len__(self) -> int:
        """Number of frames in the sequence."""
        return int(self.poses.shape[0])

    def __getitem__(self, index):
        """Return one frame's ``(17, 3)`` joints, or a sub-sequence for a slice."""
        if isinstance(index, slice):
            return Pose3DResult(poses=self.poses[index], fps=self.fps,
                                view=self.view, meta=self.meta)
        return self.poses[index]

    def __iter__(self):
        """Iterate frames, yielding ``(17, 3)`` arrays."""
        return iter(self.poses)

    def by_name(self, name: str) -> np.ndarray:
        """Return one joint's trajectory over the whole sequence.

        Args:
            name (str): A Human3.6M joint name, e.g. ``"left_wrist"``, ``"root"``.

        Returns:
            np.ndarray: ``(N, 3)`` positions of that joint across frames.

        Raises:
            KeyError: If ``name`` is not a Human3.6M joint. The message lists the
                valid names.
        """
        from .pose.config import HUMAN26M_NAMES

        joint_id = HUMAN26M_NAMES.get(name)
        if joint_id is None or joint_id >= 17:
            valid = sorted(k for k, v in HUMAN26M_NAMES.items() if v < 17)
            raise KeyError(f"{name!r} is not a Human3.6M joint. Valid names: {valid}")
        return self.poses[:, joint_id, :]

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Serialize to a JSON-safe dictionary.

        Args:
            include_arrays (bool, optional): Include the full ``(N, 17, 3)`` array as
                nested lists. Defaults to ``False``, since a long sequence is large;
                ``shape`` still records what was produced.

        Returns:
            dict: ``"task"`` (``"pose3d"``), ``"shape"``, ``"joint_order"``
                (``"human36m"``), ``"units"``, ``"fps"``, ``"view"``, ``"meta"``, and
                ``"poses"`` when requested.
        """
        out: Dict[str, Any] = {
            "task": "pose3d",
            "shape": list(self.poses.shape),
            "joint_order": "human36m",
            "units": self.meta.units if self.meta else {"poses": "relative"},
            "fps": self.fps,
            "view": getattr(self.view, "name", self.view),
            "meta": self.meta.to_dict() if self.meta else None,
        }
        if include_arrays:
            out["poses"] = self.poses.tolist()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pose3DResult":
        """Rebuild a 3D pose sequence from :meth:`to_dict` output.

        Args:
            data (dict): A serialized 3D pose result.

        Returns:
            Pose3DResult: The reconstructed sequence.

        Raises:
            KeyError: If ``data`` carries no ``"poses"`` array, which happens when it
                was serialized with ``include_arrays=False``.
        """
        if "poses" not in data:
            raise KeyError(
                "Serialized 3D pose result has no 'poses' array (it was written with "
                "include_arrays=False), so the sequence cannot be restored. "
                "Re-serialize with include_arrays=True to round-trip."
            )
        meta = ResultMeta.from_dict(data["meta"]) if data.get("meta") else None
        return cls(poses=np.asarray(data["poses"], dtype=float),
                   fps=data.get("fps"), view=data.get("view"), meta=meta)


# --------------------------------------------------------------------------- #
# TrackResult
# --------------------------------------------------------------------------- #
class TrackResult:
    """Multi-object tracker output: instances each carrying a persistent ``id``.

    Returned per frame by the tracker. Behaves like a sequence of tracked
    [`Instance`][physiotrack.Instance] objects (``len``, indexing, iteration) and
    exposes the active track :attr:`ids` and :attr:`boxes`. It may also carry the
    tracker's own rich overlay in ``rendered`` (used by :meth:`plot` when no frame
    is supplied) and the backend's ``raw`` target rows.

    Attributes:
        instances (list[Instance]): Tracked subjects; each has a persistent
            ``id`` and usually a ``box``.
        orig_img (np.ndarray | None): Source BGR frame ``(H, W, 3)``, or ``None``.
        rendered (np.ndarray | None): The tracker's own pre-rendered overlay
            image (boxes, trails, etc.), or ``None``.
        raw (list): Backend raw target rows, each
            ``[x1, y1, x2, y2, id, (cls), (conf)]``. Empty list when unset.

    Example:
        ```python
        import numpy as np
        import physiotrack as pt
        det = pt.Detection.Person()
        tracker = pt.Tracker(pt.TrackerConfig(tracker_type="ocsort", classes=[0]))
        res = det.predict(frame)
        # Tracker expects an (N, 6) [x1, y1, x2, y2, conf, cls] array:
        detections = np.array([[*i.box, i.confidence, i.cls] for i in res],
                              dtype=np.float32) if len(res) else np.empty((0, 6), np.float32)
        track_result = tracker.track(frame, detections)
        print(track_result.ids)              # e.g. [1, 2, 5]
        annotated = track_result.plot()      # tracker's own overlay
        ```

    See Also:
        [`Result`][physiotrack.Result]: image-task counterpart.
        [`Instance`][physiotrack.Instance]: a single tracked subject.
    """

    def __init__(self, *, instances: List[Instance],
                 orig_img: Optional[np.ndarray] = None,
                 rendered: Optional[np.ndarray] = None,
                 raw: Optional[list] = None):
        """Construct a tracker result (fields keyword-only).

        Args:
            instances (list[Instance]): Tracked subjects for this frame.
            orig_img (np.ndarray, optional): Source BGR frame ``(H, W, 3)``.
                Defaults to ``None``.
            rendered (np.ndarray, optional): Tracker's own overlay image.
                Defaults to ``None``.
            raw (list, optional): Backend raw target rows
                ``[x1, y1, x2, y2, id, (cls), (conf)]``. Defaults to ``None``
                (stored as an empty list).
        """
        self.instances = instances
        self.orig_img = orig_img
        # ``rendered`` is the tracker's own rich overlay (subject box, trails, etc.).
        self.rendered = rendered
        # ``raw`` is the backend's raw target rows: [x1,y1,x2,y2,id,(cls),(conf)].
        self.raw = raw if raw is not None else []

    def __iter__(self):
        """Iterate over the tracked instances.

        Yields:
            Instance: Each tracked [`Instance`][physiotrack.Instance].
        """
        return iter(self.instances)

    def __len__(self) -> int:
        """Return the number of active tracks.

        Returns:
            int: Count of tracked instances.
        """
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        """Return the tracked instance at the given index.

        Args:
            index (int): Zero-based track index.

        Returns:
            Instance: The [`Instance`][physiotrack.Instance] at ``index``.
        """
        return self.instances[index]

    def __repr__(self) -> str:
        return f"TrackResult(tracks={len(self.instances)})"

    @property
    def ids(self) -> List[int]:
        """Persistent track ids of all instances that have one.

        Returns:
            list[int]: Track ids, in instance order (instances without an id are
                skipped).
        """
        return [i.id for i in self.instances if i.id is not None]

    @property
    def boxes(self) -> np.ndarray:
        """Bounding boxes of all tracked instances that have one.

        Returns:
            np.ndarray: Float32 array of shape ``(M, 4)`` with rows
                ``[x1, y1, x2, y2]``; an empty ``(0, 4)`` array when there are
                none.
        """
        boxes = [i.box for i in self.instances if i.box is not None]
        return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), np.float32)

    def plot(self, frame: np.ndarray = None, *, boxes: bool = True, labels: bool = True,
             color: tuple = (255, 0, 0), thickness: int = 2) -> np.ndarray:
        """Render tracked boxes and ids onto a frame.

        If ``frame`` is omitted, returns the tracker's own ``rendered`` overlay
        when available, else draws on ``orig_img``. When a ``frame`` is given, a
        copy is annotated and returned (the input is not modified).

        Args:
            frame (np.ndarray, optional): BGR frame ``(H, W, 3)`` to draw on.
                Defaults to ``None`` (use ``rendered`` or ``orig_img``).
            boxes (bool, optional): Draw bounding boxes. Defaults to ``True``.
            labels (bool, optional): Draw ``"ID <n>"`` labels above boxes.
                Defaults to ``True``.
            color (tuple, optional): Box/label BGR color. Defaults to
                ``(255, 0, 0)`` (blue).
            thickness (int, optional): Box line thickness in pixels. Defaults to
                ``2``.

        Returns:
            np.ndarray: Annotated BGR image of shape ``(H, W, 3)``.

        Raises:
            ValueError: If no ``frame`` is supplied and neither ``rendered`` nor
                ``orig_img`` is available.
            RuntimeError: If OpenCV (``cv2``) is not installed.

        Example:
            ```python
            import numpy as np
            import physiotrack as pt
            det = pt.Detection.Person()
            tracker = pt.Tracker(pt.TrackerConfig(tracker_type="ocsort", classes=[0]))
            res = det.predict(frame)
            detections = np.array([[*i.box, i.confidence, i.cls] for i in res],
                                  dtype=np.float32) if len(res) else np.empty((0, 6), np.float32)
            tr = tracker.track(frame, detections)
            annotated = tr.plot(frame, color=(0, 255, 0))
            ```
        """
        # With no frame, return the tracker's own rich overlay if available.
        if frame is None:
            if self.rendered is not None:
                return self.rendered
            if self.orig_img is None:
                raise ValueError("No frame supplied and no rendered/orig_img available.")
            frame = self.orig_img
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is required for TrackResult.plot().")
        img = frame.copy()
        for inst in self.instances:
            if boxes and inst.box is not None:
                x1, y1, x2, y2 = [int(v) for v in inst.box]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                if labels and inst.id is not None:
                    cv2.putText(img, f"ID {inst.id}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return img

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Serialize the tracker result to a plain, JSON-friendly dict.

        Uses the same ``"instances"`` key and per-instance keys as
        [`Result.to_dict`][physiotrack.Result.to_dict], so a consumer does not need a
        separate code path for tracker output.

        Args:
            include_arrays (bool, optional): Include per-instance masks. Defaults to
                ``False``.

        Returns:
            dict: ``"task"`` (``"track"``) and ``"instances"``, each carrying its
                persistent ``id``.
        """
        return {
            "task": "track",
            "instances": [i.to_dict(include_arrays=include_arrays) for i in self.instances],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any],
                  orig_img: Optional[np.ndarray] = None) -> "TrackResult":
        """Rebuild a tracker result from :meth:`to_dict` output.

        Args:
            data (dict): A serialized tracker result.
            orig_img (np.ndarray, optional): Source frame to attach. Defaults to ``None``.

        Returns:
            TrackResult: The reconstructed result. The rendered overlay and the backend's
                raw rows are not serialized, so they are absent.
        """
        return cls(
            instances=[Instance.from_dict(d) for d in data.get("instances", [])],
            orig_img=orig_img,
        )


# --------------------------------------------------------------------------- #
# Video output: one FrameResult per frame, collected in a VideoResults sequence
# --------------------------------------------------------------------------- #
class FrameResult:
    """One frame of [`Video`][physiotrack.Video] output.

    A frame carries more than the subjects in it: the physiological signals
    (:attr:`vitals`) and the head orientations are properties of the *frame*, not of any
    one instance. ``FrameResult`` wraps the per-frame [`Result`][physiotrack.Result]
    together with those extras, so :class:`Result` stays a general per-task container
    rather than accumulating pipeline-specific fields.

    Attributes:
        result (Result): The subjects detected in this frame, with their keypoints.
        meta (ResultMeta): Frame index, timestamp and source frame rate.
        vitals (dict | None): rPPG-derived signals for this frame -- any of ``hr`` (bpm),
            ``snr`` (dB), ``hrv`` (index name to value) and ``respiration``
            (breaths/min). ``None`` when no vitals were requested.
        face_orientation (list | None): Per-face head-pose entries, each with ``box`` and
            ``orientation`` (``{"yaw", "pitch", "roll"}`` in degrees). Kept separate from
            ``result.instances`` because faces are detected independently of bodies and
            the two are not associated.
        track_box (list | None): The locked subject's box ``[x1, y1, x2, y2]`` when
            subject-lock tracking is enabled.

    Example:
        ```python
        import physiotrack as pt

        results = pt.Video(source="clip.mp4", pose=pt.Pose.Person(), rppg=True).run()
        for frame in results:
            print(frame.meta.timestamp, len(frame), frame.hr)
        ```

    See Also:
        [`VideoResults`][physiotrack.VideoResults]: the sequence of these that
            [`Video.run`][physiotrack.Video.run] returns.
    """

    __slots__ = ("result", "meta", "vitals", "face_orientation", "track_box")

    def __init__(self, *, result: Result, meta: Optional[ResultMeta] = None,
                 vitals: Optional[dict] = None,
                 face_orientation: Optional[list] = None,
                 track_box: Optional[list] = None):
        """Construct a frame result (all fields keyword-only).

        Args:
            result (Result): The per-frame instances.
            meta (ResultMeta, optional): Frame provenance. Defaults to the ``result``'s
                own metadata, so the two cannot disagree.
            vitals (dict, optional): rPPG-derived signals. Defaults to ``None``.
            face_orientation (list, optional): Per-face head poses. Defaults to ``None``.
            track_box (list, optional): Locked-subject box. Defaults to ``None``.
        """
        self.result = result
        self.meta = meta if meta is not None else result.meta
        self.vitals = vitals
        self.face_orientation = face_orientation
        self.track_box = track_box

    # -- container protocol: behave like the instances in the frame ------------ #
    def __iter__(self):
        """Iterate the frame's instances.

        Yields:
            Instance: Each subject detected in this frame.
        """
        return iter(self.result)

    def __len__(self) -> int:
        """Return the number of instances in the frame."""
        return len(self.result)

    def __getitem__(self, index):
        """Return the instance at ``index``."""
        return self.result[index]

    # -- convenience accessors ------------------------------------------------- #
    @property
    def instances(self) -> List[Instance]:
        """The frame's instances (shorthand for ``frame.result.instances``)."""
        return self.result.instances

    @property
    def hr(self) -> Optional[float]:
        """Heart rate in bpm for this frame, or ``None`` if unavailable.

        Note:
            Interpret alongside :attr:`snr`: a heart rate is reported whenever the
            analysis window is full, regardless of signal quality.
        """
        return (self.vitals or {}).get("hr")

    @property
    def snr(self) -> Optional[float]:
        """Signal-to-noise ratio in dB of the pulse signal, or ``None``."""
        return (self.vitals or {}).get("snr")

    def plot(self, **kwargs) -> np.ndarray:
        """Render the frame's overlay.

        Args:
            **kwargs (Any): Forwarded to [`Result.plot`][physiotrack.Result.plot].

        Returns:
            np.ndarray: The annotated BGR frame.
        """
        return self.result.plot(**kwargs)

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Serialize the frame to the JSON schema the pipeline writes.

        Args:
            include_arrays (bool, optional): Include masks and segmentation maps.
                Defaults to ``False``.

        Returns:
            dict: ``frame_id``, ``timestamp``, ``instances``, and whichever of
                ``track_box``, ``face_orientation`` and ``vitals`` are present.
        """
        out: Dict[str, Any] = {
            "frame_id": self.meta.frame_index,
            "timestamp": self.meta.timestamp,
        }
        if self.track_box is not None:
            out["track_box"] = self.track_box
        out["instances"] = [i.to_dict(include_arrays=include_arrays)
                            for i in self.result.instances]
        if self.face_orientation is not None:
            out["face_orientation"] = self.face_orientation
        if self.vitals is not None:
            out["vitals"] = self.vitals
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, architecture: str = "WHOLEBODY",
                  orig_img: Optional[np.ndarray] = None) -> "FrameResult":
        """Rebuild a frame from :meth:`to_dict` output.

        Args:
            data (dict): One serialized frame record.
            architecture (str, optional): Skeleton naming for the keypoints. Defaults to
                ``"WHOLEBODY"``.
            orig_img (np.ndarray, optional): Source frame to attach. Defaults to ``None``.

        Returns:
            FrameResult: The reconstructed frame.
        """
        meta = ResultMeta(frame_index=data.get("frame_id"),
                          timestamp=data.get("timestamp"))
        result = Result(
            orig_img=orig_img,
            instances=[Instance.from_dict(d, architecture)
                       for d in data.get("instances", [])],
            task="pose",
            architecture=architecture,
            meta=meta,
        )
        return cls(result=result, meta=meta, vitals=data.get("vitals"),
                   face_orientation=data.get("face_orientation"),
                   track_box=data.get("track_box"))

    def __repr__(self) -> str:
        parts = [f"frame={self.meta.frame_index}", f"instances={len(self)}"]
        if self.vitals:
            parts.append(f"vitals={sorted(self.vitals)}")
        return f"FrameResult({', '.join(parts)})"


class VideoResults(list):
    """The sequence of [`FrameResult`][physiotrack.FrameResult] a video run produces.

    A ``list`` subclass, so it indexes, slices and iterates like one, while adding the
    serialization the pipeline needs. Returning this rather than a list of plain dicts is
    what lets the object model survive video processing: every frame still exposes
    [`Instance`][physiotrack.Instance] objects with named keypoints.

    Example:
        ```python
        import physiotrack as pt

        results = pt.Video(source="clip.mp4", pose=pt.Pose.Person()).run()
        len(results)                        # number of frames
        results[0][0].keypoints.by_name("nose")
        results.to_json("out.json")
        ```

    See Also:
        [`Video.run`][physiotrack.Video.run]: produces this.
    """

    def to_dict_list(self, include_arrays: bool = False) -> List[Dict[str, Any]]:
        """Serialize every frame.

        Args:
            include_arrays (bool, optional): Include masks and segmentation maps.
                Defaults to ``False``.

        Returns:
            list[dict]: One record per frame -- the schema written to JSON and consumed by
                the ``physiotrack.signals`` sequence functions.
        """
        return [f.to_dict(include_arrays=include_arrays) for f in self]

    def to_json(self, path=None, *, include_arrays: bool = False, indent: int = 2):
        """Serialize to JSON, optionally writing it to a file.

        Args:
            path (str | os.PathLike, optional): Destination. Defaults to ``None``, which
                returns the JSON string.
            include_arrays (bool, optional): Include masks and segmentation maps.
                Defaults to ``False``.
            indent (int, optional): JSON indentation. Defaults to ``2``.

        Returns:
            str | None: The JSON string when ``path`` is ``None``, else ``None``.
        """
        import json

        payload = json.dumps(self.to_dict_list(include_arrays=include_arrays),
                             indent=indent)
        if path is None:
            return payload
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
        return None

    @classmethod
    def from_dict_list(cls, records, *,
                       architecture: str = "WHOLEBODY") -> "VideoResults":
        """Rebuild from serialized frame records.

        Args:
            records (Iterable[dict]): Frame records, e.g. ``json.load`` of a file written
                by [`to_json`][physiotrack.VideoResults.to_json].
            architecture (str, optional): Skeleton naming for the keypoints. Defaults to
                ``"WHOLEBODY"``.

        Returns:
            VideoResults: The reconstructed sequence.
        """
        return cls(FrameResult.from_dict(r, architecture=architecture) for r in records)

    def __repr__(self) -> str:
        return f"VideoResults({len(self)} frames)"
