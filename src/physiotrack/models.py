from enum import Enum
import inspect
import os
import requests
from tqdm import tqdm
from pathlib import Path

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from ._logging import get_logger
from ._paths import legacy_weights_dir, weights_dir

logger = get_logger(__name__)

try:
    # Read the installed metadata directly rather than importing the package's own
    # __init__, which would make this module depend on its parent finishing first.
    __version__ = _pkg_version("physiotrack")
except PackageNotFoundError:  # pragma: no cover - source tree that was never installed
    __version__ = "0.0.0.dev0"

# The weight cache lives in a per-user directory (see physiotrack._paths), never inside
# the installed package. ``Models.resolve()`` is the single way any loader learns where a
# checkpoint is, so this location is stated in exactly one place.
#
# Resolved per call rather than cached at import: ``PHYSIOTRACK_HOME`` must still take
# effect when it is set after ``import physiotrack``, which is what tests and notebooks do.


class Models:
    """Central registry of every pretrained model physiotrack can download and run.

    ``Models`` is a namespace of nested classes and ``Enum`` groups arranged by a
    four-level hierarchy::

        Models.<Task>.<Backend>.<Enum>.<member>

    - **Task** — what the model does: ``Detection``, ``Pose``, ``Pose3D``,
      ``Depth``, ``Segmentation``.
    - **Backend** — the architecture/family, e.g. ``YOLO``, ``RTDETR``, ``Sapiens``,
      ``ViTPose``, ``MotionBERT``, ``DDH``, ``DepthAnythingV2``, ``SegFace``.
    - **Enum** — a group of interchangeable checkpoints (often by dataset or
      variant), e.g. ``Detection.YOLO.PERSON`` or ``Pose.ViTPose.WholeBody``.
    - **member** — a single checkpoint. Each member's ``.value`` is the **weight
      filename** on disk (or a relative path); ``.name`` is the short handle.

    A few groups nest one level deeper or sit directly under the task: ``Pose3D``
    backends (``MotionBERT``, ``DDH``, ``FaceOrientation``) are ``Enum``\\ s directly
    under ``Pose3D``; ``Pose3D.Canonicalizer.Models`` holds the 3DPCNet weights; and
    ``Depth.DepthAnythingV2`` is an ``Enum`` directly under ``Depth``.

    The registry contains **only checkpoints**. The canonical viewpoint a pose is
    rotated to is a parameter, not a model, and lives in
    [`CanonicalView`][physiotrack.CanonicalView].

    Use [`list`][physiotrack.Models.list] to browse the registry,
    [`info`][physiotrack.Models.info] to describe a member, and
    [`get`][physiotrack.Models.get] to resolve a dotted path string such as
    ``"Depth.ZipDepth.base"`` — which lets a checkpoint be chosen from a config file or
    command line rather than only in Python source.

    Selecting a member does not download anything by itself. Pass a member to
    [`resolve`][physiotrack.Models.resolve] to get its local weight path, fetching it
    (mostly from the project's HuggingFace repos) only on first use; predictors do this
    for you. Members whose ``.value`` is an empty string (e.g.
    ``Canonicalizer.Models.GEOMETRIC``) are markers for weight-free / algorithmic
    paths and are not downloaded.

    Weights are cached **outside** the installed package: under ``$PHYSIOTRACK_HOME``
    when set, otherwise ``$XDG_CACHE_HOME/physiotrack`` or the platform user-cache
    directory. Point ``PHYSIOTRACK_HOME`` at a shared location to reuse one download
    across environments or containers. Installs from before 1.1 kept weights inside the
    package; [`migrate_weight_cache`][physiotrack.migrate_weight_cache] moves them.

    The ``validate_*`` static methods check that a given member belongs to the
    expected task/subclass, raising a descriptive ``ValueError`` otherwise; the
    high-level predictors use them to guard their ``model=`` argument.

    Example:
        ```python
        import physiotrack as pt
        from physiotrack import Models

        # Pick a checkpoint from the registry ...
        model = Models.Pose.ViTPose.WholeBody.s_wholebody
        print(model.value)                        # 'vitpose-s-wholebody.pth'

        # ... download its weights (cached after the first call) ...
        weights_path = Models.download_model(model)

        # ... and hand it to a predictor. `Pose` is a namespace of presets, so a
        # registry member goes through `Pose.Custom`.
        pose = pt.Pose.Custom(model=model)
        ```

    Note:
        The first ``download_model`` call for a checkpoint hits the network and may
        transfer a large file; subsequent calls reuse the cached copy. YOLO
        ``PERSON`` detection/segmentation variants and all ``Pose.YOLO`` checkpoints
        are fetched automatically by ultralytics instead, so ``download_model``
        returns ``None`` for them.

    See Also:
        [`Detection`][physiotrack.Detection], [`Pose`][physiotrack.Pose],
        [`Segmentation`][physiotrack.Segmentation], [`Depth`][physiotrack.Depth]:
        predictors that consume these registry members.
    """

    class Detection:
        class YOLO:
            class PERSON(Enum):
                m_person = "yolo11m.pt"
                l_person = "yolo11l.pt"
                n_person = "yolo11n.pt"

            class FACE(Enum):
                n_face = "yolov11n-face.pt"
                m_face = "yolov11m-face.pt"
                l_face = "yolov11l-face.pt"

            class VRFACE(Enum):
                l_vrface = "yolov12l-face.pt"

            class VR(Enum):
                m_vr = "yolo11m_vr.pt"
                l_vr = "yolo11l_vr.pt"

            class VRSTUDENT(Enum):
                m_vrstudent = "yolo11m_VRstudent.pt"
                l_vrstudent = "yolo11l_VRstudent.pt"

        class RTDETR:
            class PERSON(Enum):
                x_person = "rtdetr-x.pt"
                l_person = "rtdetr-l.pt"

            class VRSTUDENT(Enum):
                x_person = "yolo11x_RLDETR_VRstudent.pt"
                l_person = "yolo11l_RLDETR_VRstudent.pt"
                
    class Pose:
        class YOLO:
            class COCO(Enum):
                M11 = "yolo11m-pose.pt"
                L11 = "yolo11l-pose.pt"
            
        class Sapiens:
            class WholeBody(Enum):
                # COCO wholebody
                B1_TS_COCOHB = "sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2"
                B06_TS_COCOHB = "sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2"
                B03_TS_COCOHB = "sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2"
            
        class ViTPose:
            class WholeBody(Enum):
                s_wholebody = "vitpose-s-wholebody.pth"
                b_wholebody = "vitpose-b-wholebody.pth"
                l_wholebody = "vitpose-l-wholebody.pth"
                h_wholebody = "vitpose-h-wholebody.pth"

            class COCO(Enum):
                b_coco = "vitpose-b-coco.pth"
                h_coco = "vitpose-h-coco.pth"
                l_coco = "vitpose-l-coco.pth"
                s_coco = "vitpose-s-coco.pth"

    class Pose3D:
        class MotionBERT(Enum):
            mb_ft_h36m_global_lite = 'FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin'
            mb_ft_h36m = 'FT_MB_release_MB_ft_h36m/best_epoch.bin'
            # mb_ft_h36m_global = ''
            mb_train_h36m = 'MB_train_h36m/best_epoch.bin'

        class DDH(Enum):
            best = 'best_epoch_DDHPose.bin'

        class FaceOrientation(Enum):
            default = '6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth'
            VR = 'CMVS-FO-VR_epoch80.pth'

        class Canonicalizer:
            class Models(Enum):
                _3DPCNetS2 = 'best_model_3DPCNetS2.pth'
                _3DPCNetS3 = 'best_model_3DPCNetS3.pth'
                _3DPCNetTC48_byCam = 'best_model_3DPCNetTC48_byCam.pth'
                _3DPCNetTC48_byAction = 'best_model_3DPCNetTC48_byAction.pth'
                GEOMETRIC = ''

    class Depth:
        class DepthAnythingV2(Enum):
            vits = "depth_anything_v2_vits.pth"
            vitb = "depth_anything_v2_vitb.pth"
            vitl = "depth_anything_v2_vitl.pth"

        class ZipDepth(Enum):
            # Lightweight monocular depth. Both checkpoints share the same
            # variant='base'/global_mode='balanced' encoder+decoder weights and
            # differ only in the upsampling head.
            base = "zipdepth_base.pth"          # GPU/server head (convex unfold)
            npu = "zipdepth_base_npu.pth"       # NPU/CPU/mobile-friendly head

        # DepthAnythingV2 architecture config per encoder type. ``input_size`` is
        # the default square inference resolution for the encoder.
        MODEL_CONFIGS = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'input_size': 518},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'input_size': 518},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'input_size': 518},
        }

        # ZipDepth build config per variant. ``upsample_unfold`` selects the head
        # matching each checkpoint; ``input_size`` is the shorter-side resolution
        # (aspect ratio preserved) the model was trained at.
        ZIPDEPTH_CONFIGS = {
            'base': {'variant': 'base', 'global_mode': 'balanced', 'upsample_unfold': True, 'input_size': 384},
            'npu': {'variant': 'base', 'global_mode': 'balanced', 'upsample_unfold': False, 'input_size': 384},
        }

    class Segmentation:
        class Sapiens:
            class BodyPart(Enum):
                B1_TS_SEG = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
                B06_TS_SEG = "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"
                B03_TS_SEG = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"

        class YOLO:
            class VRHEAD(Enum):
                M11 = "yolo11m_VR_head.pt"
                M8_251029 =  'yolo8m_VR_head_251029.pt'

            class PERSON(Enum):
                m_person = "yolo11m-seg.pt"
                l_person = "yolo11l-seg.pt"

        class SegFace:
            # Face-part parsing (CelebAMask-HQ, 19 classes). Swin-Base @ 512.
            class Face(Enum):
                swinb_celeba_512 = "segface_swinb_celeba_512.pt"


    class Face:
        class MediaPipe:
            class Landmarks(Enum):
                face_landmarker = "mediapipe/face_landmarker.task"


    # Task namespaces walked by the introspection helpers below. Kept in one place so
    # `list`, `info`, `get` and `_get_model_info` cannot disagree about what the
    # registry contains.
    _TASKS = ('Detection', 'Pose', 'Segmentation', 'Pose3D', 'Depth', 'Face')

    @staticmethod
    def _walk():
        """Yield every registry member with its position in the tree.

        Walks the nested ``Task.Backend[.Group].member`` structure once, so callers do
        not have to know how deeply any particular task nests its enums.

        Yields:
            tuple[str, str, str, enum.Enum]: ``(task, backend, group, member)`` where
                ``group`` is the enum class name (equal to ``backend`` for tasks whose
                enums sit directly under the backend).
        """
        for task in Models._TASKS:
            namespace = getattr(Models, task, None)
            if namespace is None:
                continue
            for backend_name in dir(namespace):
                if backend_name.startswith('_'):
                    continue
                backend = getattr(namespace, backend_name)
                if not inspect.isclass(backend):
                    continue
                if issubclass(backend, Enum):
                    # Enums sitting directly under the task (e.g. Depth.ZipDepth).
                    for member in backend:
                        yield task, backend_name, backend_name, member
                    continue
                for group_name in dir(backend):
                    if group_name.startswith('_'):
                        continue
                    group = getattr(backend, group_name)
                    if inspect.isclass(group) and issubclass(group, Enum):
                        for member in group:
                            yield task, backend_name, group_name, member

    @staticmethod
    def path_of(member):
        """Return the dotted registry path of a member.

        Args:
            member (enum.Enum): A registry member.

        Returns:
            str | None: The dotted path, e.g.
                ``"Pose.ViTPose.WholeBody.s_wholebody"``, or ``None`` if the member is
                not part of the registry.
        """
        for task, backend, group, candidate in Models._walk():
            if candidate is member:
                prefix = f"{task}.{backend}" if backend == group else f"{task}.{backend}.{group}"
                return f"{prefix}.{candidate.name}"
        return None

    @staticmethod
    def list(task=None, backend=None, weights_only=False):
        """List the registry as dotted paths.

        Args:
            task (str, optional): Restrict to one task — ``"Detection"``, ``"Pose"``,
                ``"Segmentation"``, ``"Pose3D"`` or ``"Depth"`` (case-insensitive).
                Defaults to ``None`` (all tasks).
            backend (str, optional): Restrict to one backend, e.g. ``"ViTPose"``
                (case-insensitive). Defaults to ``None`` (all backends).
            weights_only (bool, optional): Skip members that carry no weight file —
                currently just the training-free ``Canonicalizer.Models.GEOMETRIC``
                marker. Defaults to ``False``.

        Returns:
            list[str]: Sorted dotted paths, each accepted by
                [`get`][physiotrack.Models.get].

        Raises:
            ValueError: If ``task`` is not a known task name.

        Example:
            ```python
            from physiotrack import Models
            Models.list(task="depth")
            # ['Depth.DepthAnythingV2.vitb', 'Depth.DepthAnythingV2.vitl', ...]
            len(Models.list(weights_only=True))   # 51 downloadable checkpoints
            ```
        """
        if task is not None:
            match = {t.lower(): t for t in Models._TASKS}.get(task.lower())
            if match is None:
                raise ValueError(
                    f"Unknown task {task!r}. Valid tasks: {', '.join(Models._TASKS)}."
                )
            task = match

        out = []
        for t, b, g, member in Models._walk():
            if task is not None and t != task:
                continue
            if backend is not None and b.lower() != backend.lower():
                continue
            if weights_only and not member.value:
                continue
            prefix = f"{t}.{b}" if b == g else f"{t}.{b}.{g}"
            out.append(f"{prefix}.{member.name}")
        return sorted(out)

    @staticmethod
    def get(path):
        """Resolve a dotted registry path to its member.

        Lets a model be named as a plain string, so a checkpoint can be selected from
        a config file, a CLI argument or an environment variable rather than only in
        Python source.

        Args:
            path (str): Dotted path as returned by [`list`][physiotrack.Models.list],
                e.g. ``"Pose.ViTPose.WholeBody.s_wholebody"``.

        Returns:
            enum.Enum: The matching registry member.

        Raises:
            ValueError: If no member has that path. The message suggests the closest
                available paths.

        Example:
            ```python
            import physiotrack as pt
            model = pt.Models.get("Depth.ZipDepth.base")
            depth = pt.Depth.Custom(model=model)
            ```
        """
        wanted = str(path).strip()
        for candidate in Models._walk():
            member = candidate[3]
            if Models.path_of(member) == wanted:
                return member

        tail = wanted.rsplit('.', 1)[-1].lower()
        near = [p for p in Models.list() if tail and tail in p.lower()][:5]
        hint = f" Closest matches: {', '.join(near)}." if near else ""
        raise ValueError(f"No registry model at path {path!r}.{hint}")

    @staticmethod
    def info(member):
        """Describe a registry member.

        Args:
            member (enum.Enum): A registry member.

        Returns:
            dict: ``task``, ``backend``, ``group``, ``name``, ``file_name`` (the weight
                file, empty for weight-free markers), ``path`` (the dotted path) and
                ``has_weights``.

        Raises:
            ValueError: If ``member`` is not a registry member.

        Example:
            ```python
            from physiotrack import Models
            Models.info(Models.Depth.ZipDepth.base)
            # {'task': 'Depth', 'backend': 'ZipDepth', ..., 'has_weights': True}
            ```
        """
        for task, backend, group, candidate in Models._walk():
            if candidate is member:
                prefix = f"{task}.{backend}" if backend == group else f"{task}.{backend}.{group}"
                return {
                    'task': task,
                    'backend': backend,
                    'group': group,
                    'name': candidate.name,
                    'file_name': candidate.value,
                    'path': f"{prefix}.{candidate.name}",
                    'has_weights': bool(candidate.value),
                }
        raise ValueError(f"{member!r} is not a member of the Models registry.")

    @staticmethod
    def _get_model_info(model_enum):
        """Extract model information from enum instance"""
        if not isinstance(model_enum, Enum):
            return None
            
        for category_name in ['Detection', 'Pose', 'Segmentation', 'Pose3D', 'Depth', 'Face']:
            category = getattr(Models, category_name, None)
            if not category:
                continue
            for backend_name in dir(category):
                if backend_name.startswith('_'):
                    continue
                    
                backend = getattr(category, backend_name)
                if not inspect.isclass(backend):
                    continue
                if category_name == "Pose3D":
                    if issubclass(backend, Enum) and isinstance(model_enum, backend):
                        return {
                            'category': category_name,
                            'backend': backend_name,
                            'enum_class': backend_name,  # For Pose3D, backend and enum_class are the same
                            'model_name': model_enum.name,
                            'file_name': model_enum.value
                        }
                    # Check for Canonicalizer models
                    elif backend_name == 'Canonicalizer':
                        for enum_class_name in dir(backend):
                            if enum_class_name.startswith('_'):
                                continue
                            enum_class = getattr(backend, enum_class_name)
                            if (inspect.isclass(enum_class) and
                                issubclass(enum_class, Enum) and
                                isinstance(model_enum, enum_class)):
                                return {
                                    'category': category_name,
                                    'backend': 'Canonicalizer',
                                    'enum_class': enum_class_name,
                                    'model_name': model_enum.name,
                                    'file_name': model_enum.value
                                }
                elif category_name == "Depth":
                    # Depth has enums directly under the category (e.g., Depth.DepthAnythingV2)
                    if issubclass(backend, Enum) and isinstance(model_enum, backend):
                        return {
                            'category': category_name,
                            'backend': backend_name,  # e.g., 'DepthAnythingV2'
                            'enum_class': backend_name,
                            'model_name': model_enum.name,  # e.g., 'vitl'
                            'file_name': model_enum.value  # e.g., 'depth_anything_v2_vitl.pth'
                        }
                else:
                    for enum_class_name in dir(backend):
                        if enum_class_name.startswith('_'):
                            continue
                        enum_class = getattr(backend, enum_class_name)
                        if (inspect.isclass(enum_class) and 
                            issubclass(enum_class, Enum) and 
                            isinstance(model_enum, enum_class)):
                            return {
                                'category': category_name,
                                'backend': backend_name,
                                'enum_class': enum_class_name,
                                'model_name': model_enum.name,
                                'file_name': model_enum.value
                            }
        return None
    
    @staticmethod
    def _download_yolo_model(model_info, download_path):
        """Download ViTPose models from HuggingFace"""
        file_name = model_info['file_name']
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_sapiens_model(model_info, download_path):
        """Download Sapiens models from HuggingFace"""
        file_name = model_info['file_name']

        parts = file_name.split('_')
        size = parts[1] if len(parts) > 1 else "1b"

        size_map = {"03b": "0.3b", "06b": "0.6b", "1b": "1b"}
        size = size_map.get(size, size)

        if model_info['category'] == 'Pose':
            task = "pose-coco"
            format_type = "torchscript"
            base_url = f"https://huggingface.co/noahcao/sapiens-{task}/resolve/main/sapiens_lite_host/{format_type}/pose/checkpoints/sapiens_{size}"
        elif model_info['category'] == 'Segmentation':
            # Sapiens segmentation models - all use facebook repos
            task = "seg"
            format_type = "torchscript"
            base_url = f"https://huggingface.co/facebook/sapiens-{task}-{size}-{format_type}/resolve/main"
        else:
            raise ValueError(
                f"Sapiens weights are only hosted for the Pose and Segmentation tasks, "
                f"got category {model_info['category']!r} for {file_name!r}."
            )
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_vitpose_model(model_info, download_path):
        """Download ViTPose models from HuggingFace"""
        file_name = model_info['file_name']
        dataset = model_info['enum_class'].lower()  # 'wholebody' or 'coco'
        base_url = f"https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/{dataset}"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)
    
    @staticmethod
    def _download_motionbert_model(model_info, download_path):
        """Download MotionBERT models from HuggingFace"""
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        actual_filename = os.path.basename(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)
        base_url = f"https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d"
        download_url = f"{base_url}/{file_name}?download=true"
        
        return Models._download_file(download_url, actual_filename, full_download_path)

    @staticmethod
    def _download_ddh_model(model_info, download_path):
        """Download DDHPose models from HuggingFace"""
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        actual_filename = os.path.basename(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        
        return Models._download_file(download_url, actual_filename, full_download_path)
    
    @staticmethod
    def _download_canonicalizer_model(model_info, download_path):
        """Download a Canonicalizer (3DPCNet) checkpoint from HuggingFace.

        All 3DPCNet checkpoints share one architecture and load from a single
        bundled inference config, so only the ``.pth`` weights are downloaded.
        """
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"

        model_download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(model_download_url, file_name, full_download_path)

    @staticmethod
    def _download_depth_model(model_info, download_path):
        """Download DepthAnythingV2 models from tharindu326/physiotrack HuggingFace repo"""
        file_name = model_info['file_name']
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_zipdepth_model(model_info, download_path):
        """Download a ZipDepth checkpoint from the tharindu326/physiotrack HuggingFace repo."""
        file_name = model_info['file_name']
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_segface_model(model_info, download_path):
        """Download a SegFace face-parsing checkpoint from the physiotrack HuggingFace repo."""
        file_name = model_info['file_name']
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_mediapipe_model(model_info, download_path):
        """Download the MediaPipe Face Landmarker model bundle."""
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        actual_filename = os.path.basename(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)

        download_url = (
            "https://storage.googleapis.com/"
            "mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/"
            "face_landmarker.task"
        )

        return Models._download_file(
            download_url,
            actual_filename,
            full_download_path,
        )

    @staticmethod
    def _download_file(url, file_name, download_path):
        """Download a weight file into the cache, atomically.

        The download streams to a temporary sibling file and is renamed into place
        only once complete, so an interrupted transfer can never leave a truncated
        checkpoint that later loads as a corrupt model.

        Args:
            url (str): Source URL.
            file_name (str): Destination file name within ``download_path``.
            download_path (str | os.PathLike): Destination directory.

        Returns:
            str: Absolute path to the cached file.

        Raises:
            requests.exceptions.RequestException: If the transfer fails.
        """
        os.makedirs(download_path, exist_ok=True)
        file_path = os.path.join(download_path, file_name)

        if os.path.exists(file_path):
            return file_path

        # Identify the client honestly; some hosts reject an empty User-Agent.
        headers = {"User-Agent": f"physiotrack/{__version__} (+https://github.com/tharindu326/physiotrack)"}
        tmp_path = f"{file_path}.part"

        try:
            response = requests.get(url, stream=True, headers=headers, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            with tqdm(total=total_size, unit="iB", unit_scale=True, desc=file_name) as pbar:
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
                            pbar.update(len(chunk))

            if total_size and os.path.getsize(tmp_path) != total_size:
                raise IOError(
                    f"Incomplete download for {file_name}: expected {total_size} bytes, "
                    f"got {os.path.getsize(tmp_path)}."
                )

            os.replace(tmp_path, file_path)
            return file_path

        except BaseException:
            # Covers KeyboardInterrupt as well as request failures: never leave a
            # partial file behind for the exists() check above to trust later.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @staticmethod
    def resolve(model_enum):
        """Return the local path to a registry member's weights, fetching on first use.

        This is the only place in the library that knows where weights live. Every
        predictor and every backend loader calls it instead of deriving a path, so the
        cache location is a single fact rather than something restated per module, and
        the download-if-missing check cannot drift between call sites.

        Args:
            model_enum (enum.Enum): A registry member, e.g.
                ``Models.Pose.ViTPose.WholeBody.s_wholebody``.

        Returns:
            str | None: Absolute path to the cached weight file, or ``None`` for
                members that ultralytics fetches on demand (any ``Pose.YOLO`` model
                and any ``PERSON`` YOLO/RTDETR variant) and for weight-free markers
                such as the geometric canonicalizer.

        Raises:
            ValueError: If ``model_enum`` is not a registry member.
            requests.exceptions.RequestException: If a required download fails.

        Example:
            ```python
            from physiotrack import Models
            path = Models.resolve(Models.Depth.ZipDepth.base)
            ```

        Note:
            Weights are cached under ``$PHYSIOTRACK_HOME`` when set, otherwise in the
            platform user-cache directory — never inside ``site-packages``. Set
            ``PHYSIOTRACK_HOME`` to share one cache between environments.
        """
        if not isinstance(model_enum, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model_enum)}")
        if not model_enum.value:
            return None  # weight-free registry marker (e.g. GEOMETRIC canonicalizer)

        path = os.path.join(str(weights_dir()), model_enum.value)
        if os.path.isfile(path):
            return path

        # Weights are only ever read from the cache. If a pre-1.1 install already put
        # this checkpoint inside the package, say so before spending the bandwidth --
        # re-downloading several GB the user already has is worse than one log line.
        stale = legacy_weights_dir() / model_enum.value
        if stale.is_file():
            logger.info(
                "%s already exists in the pre-1.1 in-package location (%s) but weights "
                "now live in %s. Run physiotrack.migrate_weight_cache() to move them "
                "instead of re-downloading.", model_enum.value, stale.parent, os.path.dirname(path))

        downloaded = Models.download_model(model_enum)
        if downloaded is None:
            return None  # ultralytics fetches this one itself
        return downloaded

    @staticmethod
    def download_model(model_enum, download_path=None):
        """Download a registry model's weights and return the local file path.

        Resolves which task/backend the member belongs to, then fetches the weight
        file from the appropriate HuggingFace repository, showing a progress bar. If
        the file already exists at the destination it is reused (no re-download).

        Args:
            model_enum (enum.Enum): A registry member, e.g.
                ``Models.Pose.ViTPose.WholeBody.s_wholebody`` or
                ``Models.Depth.DepthAnythingV2.vitl``.
            download_path (str, optional): Directory to download into. Defaults to
                ``None``, meaning the per-user weight cache resolved by
                [`physiotrack._paths.weights_dir`][] and overridable with
                ``$PHYSIOTRACK_HOME``. For members whose value contains a
                subdirectory (e.g. MotionBERT), that subdirectory is created under
                this path. Prefer [`Models.resolve`][physiotrack.Models.resolve],
                which reuses an already-cached file instead of re-resolving the path.

        Returns:
            str | None: Absolute path to the downloaded (or cached) weight file, or
                ``None`` for backends handled elsewhere — specifically any
                ``Pose.YOLO`` model and any ``PERSON`` YOLO/RTDETR variant, which
                ultralytics downloads on demand.

        Raises:
            ValueError: If ``model_enum`` is not an ``Enum`` instance, cannot be
                located in the registry, or belongs to an unknown backend.
            requests.exceptions.RequestException: If the HTTP download fails (a
                partial file is removed before re-raising).

        Example:
            ```python
            from physiotrack import Models
            path = Models.download_model(Models.Pose.Sapiens.WholeBody.B03_TS_COCOHB)
            ```

        Note:
            The first call transfers the full checkpoint over the network; later
            calls for the same file return the cached path immediately.
        """
        if not isinstance(model_enum, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model_enum)}")
        if download_path is None:
            download_path = str(weights_dir())

        model_info = Models._get_model_info(model_enum)
        if not model_info:
            raise ValueError(f"Could not determine model information for {model_enum}")
        
        # print(f"Downloading {model_info['category']} model: {model_info['backend']}.{model_info['enum_class']}.{model_info['model_name']}")
        if model_info['backend'] in ('YOLO', 'RTDETR'):
            # Pose-YOLO and any PERSON variant (detection or segmentation) auto-download
            # via ultralytics; everything else (FACE/VR/VRSTUDENT/VRHEAD/...) is hosted.
            if model_info['category'] == 'Pose' or model_info['enum_class'] == 'PERSON':
                return None
            return Models._download_yolo_model(model_info, download_path)
        elif model_info['backend'] == 'Sapiens':
            return Models._download_sapiens_model(model_info, download_path)
        elif model_info['backend'] == 'ViTPose':
            return Models._download_vitpose_model(model_info, download_path)
        elif model_info['backend'] == 'MotionBERT':
            return Models._download_motionbert_model(model_info, download_path)
        elif model_info['backend'] == 'DDH':
            return Models._download_ddh_model(model_info, download_path)
        elif model_info['backend'] == 'Canonicalizer':
            return Models._download_canonicalizer_model(model_info, download_path)
        elif model_info['backend'] == 'FaceOrientation':
            # FaceOrientation uses HuggingFace download like DDH
            return Models._download_ddh_model(model_info, download_path)
        elif model_info['backend'] == 'DepthAnythingV2':
            return Models._download_depth_model(model_info, download_path)
        elif model_info['backend'] == 'ZipDepth':
            return Models._download_zipdepth_model(model_info, download_path)
        elif model_info['backend'] == 'SegFace':
            return Models._download_segface_model(model_info, download_path)
        elif model_info['backend'] == 'MediaPipe':
            return Models._download_mediapipe_model(model_info, download_path)
        else:
            raise ValueError(f"Unknown backend: {model_info['backend']}")

    @staticmethod
    def validate_det_model(model, expected_subclass: str = None):
        """Verify a detection model is valid, optionally for a specific subclass.

        With ``expected_subclass`` given, checks ``model`` against the enum of that
        name under ``Models.Detection.YOLO`` or ``Models.Detection.RTDETR`` (matched
        case-insensitively). Without it, accepts ``model`` if it is a member of any
        enum under any ``Models.Detection`` backend — which is what
        ``Detection.Custom`` needs, since it deliberately imposes no subclass.
        Returns ``None`` on success.

        Args:
            model (enum.Enum): The candidate detection registry member, e.g.
                ``Models.Detection.YOLO.PERSON.m_person``.
            expected_subclass (str, optional): Name of the required enum group, e.g.
                ``"PERSON"``, ``"FACE"``, ``"VR"``, ``"VRSTUDENT"``, ``"VRFACE"``
                (case-insensitive). Defaults to ``None`` (accept any detection model).

        Raises:
            ValueError: If ``model`` is not an ``Enum``, if no subclass named
                ``expected_subclass`` exists in YOLO or RTDETR, or if ``model`` is
                not a member of that subclass (the message lists valid members).

        Example:
            ```python
            from physiotrack import Models
            Models.validate_det_model(Models.Detection.YOLO.PERSON.m_person, "PERSON")
            Models.validate_det_model(Models.Detection.YOLO.FACE.m_face)  # any subclass
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum member for `model`, got {type(model).__name__}")

        if expected_subclass is None:
            for backend_name in dir(Models.Detection):
                if backend_name.startswith("_"):
                    continue
                backend = getattr(Models.Detection, backend_name)
                if not inspect.isclass(backend):
                    continue
                for enum_class_name in dir(backend):
                    if enum_class_name.startswith("_"):
                        continue
                    enum_class = getattr(backend, enum_class_name)
                    if (inspect.isclass(enum_class)
                            and issubclass(enum_class, Enum)
                            and isinstance(model, enum_class)):
                        return  # valid
            raise ValueError(
                f"Invalid detection model: {repr(model)}.\n"
                f"Expected a valid enum member from Models.Detection.<Backend>.<EnumClass>"
            )

        target = expected_subclass.strip().upper()
        enum_classes = []
        for backend in (Models.Detection.YOLO, Models.Detection.RTDETR):
            if hasattr(backend, target):
                enum_classes.append(getattr(backend, target))
        if not enum_classes:
            raise ValueError(f"No detection subclass named '{expected_subclass}' in YOLO or RTDETR.")
        for enum_cls in enum_classes:
            if isinstance(model, enum_cls):
                return  # ✅ valid
        all_valid = []
        for enum_cls in enum_classes:
            names = ", ".join(e.name for e in enum_cls)
            all_valid.append(f"{enum_cls.__module__.split('.')[-1]}.{enum_cls.__name__}: [{names}]")
        valid_str = "\n  ".join(all_valid)
        raise ValueError(
            f"Model '{model.name}' is not valid for subclass '{expected_subclass}'.\n"
            f"Valid members are:\n  {valid_str}"
        )

    @staticmethod
    def validate_seg_model(model, expected_subclass: str = None):
        """Verify a segmentation model is valid, optionally for a specific subclass.

        With ``expected_subclass`` given, checks ``model`` against the enum of that
        name under ``Models.Segmentation.YOLO`` or ``Models.Segmentation.Sapiens``.
        Without it, accepts ``model`` if it is a member of any enum under any
        ``Models.Segmentation`` backend. Returns ``None`` on success.

        Args:
            model (enum.Enum): The candidate segmentation registry member, e.g.
                ``Models.Segmentation.YOLO.PERSON.m_person``.
            expected_subclass (str, optional): Name of the required enum group, e.g.
                ``"PERSON"``, ``"VRHEAD"``, ``"BodyPart"`` (matched
                case-insensitively). Defaults to ``None`` (any segmentation model
                accepted).

        Raises:
            ValueError: If ``model`` is not an ``Enum``, if ``expected_subclass`` is
                given but not found in YOLO or Sapiens, or if ``model`` is not a
                valid segmentation member.

        Example:
            ```python
            from physiotrack import Models
            Models.validate_seg_model(Models.Segmentation.Sapiens.BodyPart.B03_TS_SEG)
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum member for `model`, got {type(model).__name__}")

        # If expected_subclass is provided, validate against specific subclass
        if expected_subclass:
            target = expected_subclass.strip().upper()
            enum_classes = []
            for backend in (Models.Segmentation.YOLO, Models.Segmentation.Sapiens):
                # Check if the target exists in the backend
                for attr_name in dir(backend):
                    if attr_name.upper() == target:
                        enum_classes.append(getattr(backend, attr_name))

            if not enum_classes:
                raise ValueError(f"No segmentation subclass named '{expected_subclass}' in YOLO or Sapiens.")

            for enum_cls in enum_classes:
                if isinstance(model, enum_cls):
                    return  # ✅ valid

            all_valid = []
            for enum_cls in enum_classes:
                names = ", ".join(e.name for e in enum_cls)
                all_valid.append(f"{enum_cls.__module__.split('.')[-1]}.{enum_cls.__name__}: [{names}]")
            valid_str = "\n  ".join(all_valid)
            raise ValueError(
                f"Model '{model.name}' is not valid for subclass '{expected_subclass}'.\n"
                f"Valid members are:\n  {valid_str}"
            )
        else:
            # General validation - check if it's any valid segmentation model
            for backend_name in dir(Models.Segmentation):
                if backend_name.startswith('_'):
                    continue
                backend = getattr(Models.Segmentation, backend_name)
                if not inspect.isclass(backend):
                    continue

                for enum_class_name in dir(backend):
                    if enum_class_name.startswith('_'):
                        continue
                    enum_class = getattr(backend, enum_class_name)
                    if (inspect.isclass(enum_class) and
                        issubclass(enum_class, Enum) and
                        isinstance(model, enum_class)):
                        return  # ✅ valid

            raise ValueError(
                f"Invalid segmentation model: {repr(model)}.\n"
                f"Expected a valid enum member from Models.Segmentation.<Backend>.<EnumClass>"
            )

    @staticmethod
    def validate_pose_model(model):
        """Verify a model is a valid 2D pose registry member.

        Accepts ``model`` if it is a member of any enum under any
        ``Models.Pose`` backend (``YOLO``, ``Sapiens``, ``ViTPose``). Returns
        ``None`` on success.

        Args:
            model (enum.Enum): The candidate pose registry member, e.g.
                ``Models.Pose.ViTPose.WholeBody.s_wholebody``.

        Raises:
            ValueError: If ``model`` is not an ``Enum`` or is not a member of any
                ``Models.Pose.<Backend>.<EnumClass>``.

        Example:
            ```python
            from physiotrack import Models
            Models.validate_pose_model(Models.Pose.YOLO.COCO.M11)
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model)}")
            
        for attr_name in dir(Models.Pose):
            if attr_name.startswith('_'):
                continue
                
            backend = getattr(Models.Pose, attr_name)
            if not inspect.isclass(backend):
                continue
                
            for sub_attr_name in dir(backend):
                if sub_attr_name.startswith('_'):
                    continue
                    
                sub = getattr(backend, sub_attr_name)
                if (inspect.isclass(sub) and 
                    issubclass(sub, Enum) and 
                    isinstance(model, sub)):
                    return  # ✅ Valid model found
                    
        raise ValueError(
            f"Invalid pose model: {repr(model)}.\n"
            f"Expected a valid enum member from Models.Pose.<Backend>.<EnumClass>"
        )

    @staticmethod
    def validate_pose3d_model(model, expected_subclass=None):
        """Verify a model is a valid 3D-pose registry member.

        Accepts ``model`` if it is a member of any enum under ``Models.Pose3D``
        (including the nested ``Canonicalizer`` groups). When ``expected_subclass``
        is given, also requires ``model``'s enum class name to match it exactly.
        Returns ``None`` on success.

        Args:
            model (enum.Enum): The candidate 3D-pose registry member, e.g.
                ``Models.Pose3D.MotionBERT.mb_ft_h36m``.
            expected_subclass (str, optional): Backend/enum-class name the model must
                come from, e.g. ``"MotionBERT"``, ``"DDH"``, ``"FaceOrientation"``.
                Defaults to ``None`` (any Pose3D member accepted).

        Raises:
            ValueError: If ``model`` is not an ``Enum``, if its class name does not
                match ``expected_subclass``, or if it is not a valid Pose3D member
                (the message lists all valid members).

        Example:
            ```python
            from physiotrack import Models
            Models.validate_pose3d_model(
                Models.Pose3D.MotionBERT.mb_ft_h36m, "MotionBERT"
            )
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model)}")
        
        # If expected_subclass is provided, validate it matches
        if expected_subclass:
            model_class_name = model.__class__.__name__
            if model_class_name != expected_subclass:
                raise ValueError(
                    f"Expected model from Models.Pose3D.{expected_subclass}, "
                    f"but got {model_class_name}"
                )
            
        for attr_name in dir(Models.Pose3D):
            if attr_name.startswith('_'):
                continue
                
            backend = getattr(Models.Pose3D, attr_name)
            if not inspect.isclass(backend):
                continue
                
            # Check if this backend is an Enum class itself
            if issubclass(backend, Enum) and isinstance(model, backend):
                return  # ✅ Valid model found
                
            # Check sub-classes within the backend
            for sub_attr_name in dir(backend):
                if sub_attr_name.startswith('_'):
                    continue
                    
                sub = getattr(backend, sub_attr_name)
                if (inspect.isclass(sub) and 
                    issubclass(sub, Enum) and 
                    isinstance(model, sub)):
                    return  # ✅ Valid model found
                    
        # If we reach here, the model is not valid
        valid_models = []
        for attr_name in dir(Models.Pose3D):
            if attr_name.startswith('_'):
                continue
            backend = getattr(Models.Pose3D, attr_name)
            if inspect.isclass(backend) and issubclass(backend, Enum):
                for member in backend:
                    valid_models.append(f"Models.Pose3D.{attr_name}.{member.name}")
                    
        valid_str = "\n  ".join(valid_models)
        raise ValueError(
            f"Invalid pose3d model: {repr(model)}.\n"
            f"Expected a valid enum member from Models.Pose3D.<Backend>.<model_name>\n"
            f"Valid models are:\n  {valid_str}"
        )

    @staticmethod
    def validate_depth_model(model):
        """Verify a model is a valid depth registry member.

        Accepts ``model`` if it is a member of any enum under ``Models.Depth``
        (``DepthAnythingV2`` or ``ZipDepth``). Returns ``None`` on success.

        Args:
            model (enum.Enum): The candidate depth registry member, e.g.
                ``Models.Depth.DepthAnythingV2.vitl``.

        Raises:
            ValueError: If ``model`` is not an ``Enum`` or is not a valid
                ``Models.Depth.<Backend>.<model_name>`` (the message lists valid
                members).

        Example:
            ```python
            from physiotrack import Models
            Models.validate_depth_model(Models.Depth.DepthAnythingV2.vits)
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model)}")

        # Check if model is from Depth category
        for attr_name in dir(Models.Depth):
            if attr_name.startswith('_'):
                continue

            backend = getattr(Models.Depth, attr_name)
            if not inspect.isclass(backend):
                continue

            if issubclass(backend, Enum) and isinstance(model, backend):
                return  # ✅ Valid model found

        # If we reach here, the model is not valid
        valid_models = []
        for attr_name in dir(Models.Depth):
            if attr_name.startswith('_'):
                continue
            backend = getattr(Models.Depth, attr_name)
            if inspect.isclass(backend) and issubclass(backend, Enum):
                for member in backend:
                    valid_models.append(f"Models.Depth.{attr_name}.{member.name}")

        valid_str = "\n  ".join(valid_models)
        raise ValueError(
            f"Invalid depth model: {repr(model)}.\n"
            f"Expected a valid enum member from Models.Depth.<Backend>.<model_name>\n"
            f"Valid models are:\n  {valid_str}"
        )

    @staticmethod
    def get_depth_config(model):
        """Return the build config for a depth model, dispatched by backend.

        Looks up the settings needed to construct the depth network for ``model``.
        The returned dict is backend-specific but always carries an ``input_size``
        key giving the model's default inference resolution:

        - ``DepthAnythingV2`` members return the encoder config with keys
          ``"encoder"``, ``"features"``, ``"out_channels"`` and ``"input_size"``.
        - ``ZipDepth`` members return the build config with keys ``"variant"``,
          ``"global_mode"``, ``"upsample_unfold"`` and ``"input_size"``.

        Args:
            model (Models.Depth.*): A depth registry member, e.g.
                ``Models.Depth.DepthAnythingV2.vitl`` or ``Models.Depth.ZipDepth.base``.

        Returns:
            dict: The config for the model (see above).

        Raises:
            ValueError: If ``model`` is not a recognized ``Models.Depth`` member
                or its variant is unknown.

        Example:
            ```python
            from physiotrack import Models
            cfg = Models.get_depth_config(Models.Depth.ZipDepth.base)
            ```
        """
        if isinstance(model, Models.Depth.DepthAnythingV2):
            encoder_name = model.name  # 'vits', 'vitb', or 'vitl'
            if encoder_name not in Models.Depth.MODEL_CONFIGS:
                raise ValueError(f"Unknown DepthAnythingV2 encoder: {encoder_name}")
            return Models.Depth.MODEL_CONFIGS[encoder_name]

        if isinstance(model, Models.Depth.ZipDepth):
            variant_name = model.name  # 'base' or 'npu'
            if variant_name not in Models.Depth.ZIPDEPTH_CONFIGS:
                raise ValueError(f"Unknown ZipDepth variant: {variant_name}")
            return Models.Depth.ZIPDEPTH_CONFIGS[variant_name]

        raise ValueError(
            f"Expected a Models.Depth.<Backend> enum member, got {type(model)}"
        )


if __name__ == "__main__":
    try:
        vitpose_path = Models.download_model(Models.Pose.ViTPose.WholeBody.s_wholebody)
        sapiens_path = Models.download_model(Models.Pose.Sapiens.WholeBody.B03_TS_COCOHB)
        yolo_path = Models.download_model(Models.Detection.YOLO.VRSTUDENT.m_vrstudent)
        
    except Exception as e:
        logger.error("%s", e)