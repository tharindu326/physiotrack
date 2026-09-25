"""Keypoint index-to-name maps for the pose skeletons Physiotrack supports.

This module defines the canonical ``id -> name`` (and reverse ``name -> id``)
dictionaries for every keypoint layout used across the library. When a
[`Pose`][physiotrack.Pose] backend returns raw keypoints, the id of each point
is looked up in one of these maps to attach a human-readable ``name`` to the
resulting [`Keypoint`][physiotrack.Keypoint]. That is exactly what powers
[`Keypoints.by_name`][physiotrack.Keypoints.by_name] and
[`Keypoints.by_id`][physiotrack.Keypoints.by_id]: ``by_id`` indexes the integer
key of these dicts, while ``by_name`` indexes the string value.

The active map is selected by the model's ``architecture`` string: a whole-body
model (``architecture == "WHOLEBODY"``) uses [`COCO_WHOLEBODY`](#), while a
body-only COCO model uses [`COCO`](#). Ids not present in the active map are
surfaced by ``Keypoints`` as ``"unknown_<id>"``.

Maps:
    COCO_WHOLEBODY: The 133-point COCO-WholeBody layout plus two derived
        centroids, spanning ids ``0``-``134``. The index ranges are:

        - Body: ids ``0``-``16`` (nose, eyes, ears, shoulders, elbows, wrists,
          hips, knees, ankles) — the standard COCO-17 body skeleton.
        - Feet: ids ``17``-``22`` (left/right big toe, small toe, heel).
        - Face: ids ``23``-``90`` — a 68-point face layout (jaw ``23``-``39``,
          eyebrows ``40``-``49``, nose ``50``-``58``, eyes ``59``-``70``, mouth
          ``71``-``90``).
        - Hands: ids ``91``-``132`` — 21 points per hand (left ``91``-``111``,
          right ``112``-``132``).
        - Derived centroids: ``133`` (``head_centroid``) and ``134``
          (``body_centroid``).

    COCO: The body-only COCO-17 layout, i.e. the first 17 entries of
        ``COCO_WHOLEBODY`` (ids ``0``-``16``). Used by body-only pose models.

    COCO_WHOLEBODY_NAMES / COCO_NAMES: Reverse ``name -> id`` maps of the two
        dicts above (values become keys), useful for name-based lookups.

    HALPE_KEYPOINT_DICT: The 26-point Halpe body layout (ids ``0``-``25``),
        including three derived points — ``head_top`` (17), ``neck`` (18) and
        ``pelvis_point`` (19) — used internally during COCO->Halpe conversion for
        3D lifting (see [`Pose3D`][physiotrack.pose.pose3D.Pose3D]).

    HALPE_TO_COCO_KEYPOINT_MAP: Maps Halpe ids to their COCO-WholeBody source id
        (``None`` for the three derived Halpe points that must be computed rather
        than copied).

    HUMAN26M / HUMAN26M_NAMES: The 17-joint Human3.6M layout (ids ``0``-``16``,
        ``root``-centered) and its reverse map. This is the joint order produced
        by 3D lifting models such as MotionBERT and DDHPose.

Note:
    These dictionaries are keyed as strings for the COCO maps (``"0"``,
    ``"1"``, ...) and as integers for the Halpe/Human3.6M maps, matching how each
    is consumed downstream. Do not renumber them: the ids are contractual with
    the model outputs and the drawing/skeleton code.
"""

from itertools import islice

COCO_WHOLEBODY = {
                            # Body (17 keypoints)
                            "0": "nose",
                            "1": "left_eye",
                            "2": "right_eye",
                            "3": "left_ear",
                            "4": "right_ear",
                            "5": "left_shoulder",
                            "6": "right_shoulder",
                            "7": "left_elbow",
                            "8": "right_elbow",
                            "9": "left_wrist",
                            "10": "right_wrist",
                            "11": "left_hip",
                            "12": "right_hip",
                            "13": "left_knee",
                            "14": "right_knee",
                            "15": "left_ankle",
                            "16": "right_ankle",

                            # Feet (6 keypoints)
                            "17": "left_big_toe",
                            "18": "left_small_toe",
                            "19": "left_heel",
                            "20": "right_big_toe",
                            "21": "right_small_toe",
                            "22": "right_heel",

                            # Face (68 keypoints)
                            # Using a standard 68-point face layout:
                            # Jaw (17 points): 23-39
                            "23": "face_jaw_0",
                            "24": "face_jaw_1",
                            "25": "face_jaw_2",
                            "26": "face_jaw_3",
                            "27": "face_jaw_4",
                            "28": "face_jaw_5",
                            "29": "face_jaw_6",
                            "30": "face_jaw_7",
                            "31": "face_jaw_8",
                            "32": "face_jaw_9",
                            "33": "face_jaw_10",
                            "34": "face_jaw_11",
                            "35": "face_jaw_12",
                            "36": "face_jaw_13",
                            "37": "face_jaw_14",
                            "38": "face_jaw_15",
                            "39": "face_jaw_16",
                            # Right Eyebrow (5 points): 40-44 (subject's right, iBUG 17-21)
                            "40": "face_right_eyebrow_0",
                            "41": "face_right_eyebrow_1",
                            "42": "face_right_eyebrow_2",
                            "43": "face_right_eyebrow_3",
                            "44": "face_right_eyebrow_4",
                            # Left Eyebrow (5 points): 45-49 (subject's left, iBUG 22-26)
                            "45": "face_left_eyebrow_0",
                            "46": "face_left_eyebrow_1",
                            "47": "face_left_eyebrow_2",
                            "48": "face_left_eyebrow_3",
                            "49": "face_left_eyebrow_4",
                            # Nose (9 points): 50-58
                            "50": "face_nose_bridge_0",
                            "51": "face_nose_bridge_1",
                            "52": "face_nose_bridge_2",
                            "53": "face_nose_bridge_3",
                            "54": "face_nose_tip_0",
                            "55": "face_nose_tip_1",
                            "56": "face_nose_tip_2",
                            "57": "face_nose_tip_3",
                            "58": "face_nose_tip_4",
                            # Right Eye (6 points): 59-64 (subject's right, iBUG 36-41)
                            "59": "face_right_eye_0",
                            "60": "face_right_eye_1",
                            "61": "face_right_eye_2",
                            "62": "face_right_eye_3",
                            "63": "face_right_eye_4",
                            "64": "face_right_eye_5",
                            # Left Eye (6 points): 65-70 (subject's left, iBUG 42-47)
                            "65": "face_left_eye_0",
                            "66": "face_left_eye_1",
                            "67": "face_left_eye_2",
                            "68": "face_left_eye_3",
                            "69": "face_left_eye_4",
                            "70": "face_left_eye_5",
                            # Mouth Outer (12 points): 71-82
                            "71": "face_mouth_outer_0",
                            "72": "face_mouth_outer_1",
                            "73": "face_mouth_outer_2",
                            "74": "face_mouth_outer_3",
                            "75": "face_mouth_outer_4",
                            "76": "face_mouth_outer_5",
                            "77": "face_mouth_outer_6",
                            "78": "face_mouth_outer_7",
                            "79": "face_mouth_outer_8",
                            "80": "face_mouth_outer_9",
                            "81": "face_mouth_outer_10",
                            "82": "face_mouth_outer_11",
                            # Mouth Inner (8 points): 83-90
                            "83": "face_mouth_inner_0",
                            "84": "face_mouth_inner_1",
                            "85": "face_mouth_inner_2",
                            "86": "face_mouth_inner_3",
                            "87": "face_mouth_inner_4",
                            "88": "face_mouth_inner_5",
                            "89": "face_mouth_inner_6",
                            "90": "face_mouth_inner_7",

                            # Hands (42 keypoints: 21 per hand)
                            # Left Hand (21 points): 91-111
                            "91": "left_hand_wrist",
                            "92": "left_hand_thumb_1",
                            "93": "left_hand_thumb_2",
                            "94": "left_hand_thumb_3",
                            "95": "left_hand_thumb_tip",
                            "96": "left_hand_index_1",
                            "97": "left_hand_index_2",
                            "98": "left_hand_index_3",
                            "99": "left_hand_index_tip",
                            "100": "left_hand_middle_1",
                            "101": "left_hand_middle_2",
                            "102": "left_hand_middle_3",
                            "103": "left_hand_middle_tip",
                            "104": "left_hand_ring_1",
                            "105": "left_hand_ring_2",
                            "106": "left_hand_ring_3",
                            "107": "left_hand_ring_tip",
                            "108": "left_hand_pinky_1",
                            "109": "left_hand_pinky_2",
                            "110": "left_hand_pinky_3",
                            "111": "left_hand_pinky_tip",

                            # Right Hand (21 points): 112-132
                            "112": "right_hand_wrist",
                            "113": "right_hand_thumb_1",
                            "114": "right_hand_thumb_2",
                            "115": "right_hand_thumb_3",
                            "116": "right_hand_thumb_tip",
                            "117": "right_hand_index_1",
                            "118": "right_hand_index_2",
                            "119": "right_hand_index_3",
                            "120": "right_hand_index_tip",
                            "121": "right_hand_middle_1",
                            "122": "right_hand_middle_2",
                            "123": "right_hand_middle_3",
                            "124": "right_hand_middle_tip",
                            "125": "right_hand_ring_1",
                            "126": "right_hand_ring_2",
                            "127": "right_hand_ring_3",
                            "128": "right_hand_ring_tip",
                            "129": "right_hand_pinky_1",
                            "130": "right_hand_pinky_2",
                            "131": "right_hand_pinky_3",
                            "132": "right_hand_pinky_tip",
                            "133": "head_centroid",
                            "134": "body_centroid",
                            "135": "pelvic_centroid"
                        }


COCO = dict(islice(COCO_WHOLEBODY.items(), 17))


# MediaPipe Face Mesh (Face Landmarker): 468 surface points + 10 iris points. Sides
# are the subject's own, matching MediaPipe's naming (landmark 33 is the right eye).
# Only the points the library measures get a semantic name; the rest are
# ``facemesh_<id>``. See physiotrack.signals.face for the measurement tables.
_FACEMESH_SEMANTIC = {
    1: "nose_tip", 10: "forehead", 152: "chin",
    33: "right_eye_outer", 133: "right_eye_inner",
    160: "right_eye_upper_outer", 158: "right_eye_upper_inner",
    144: "right_eye_lower_outer", 153: "right_eye_lower_inner",
    263: "left_eye_outer", 362: "left_eye_inner",
    387: "left_eye_upper_outer", 385: "left_eye_upper_inner",
    373: "left_eye_lower_outer", 380: "left_eye_lower_inner",
    468: "right_iris_center", 469: "right_iris_1", 470: "right_iris_2",
    471: "right_iris_3", 472: "right_iris_4",
    473: "left_iris_center", 474: "left_iris_1", 475: "left_iris_2",
    476: "left_iris_3", 477: "left_iris_4",
    61: "mouth_right_corner", 291: "mouth_left_corner",
    0: "upper_lip_outer", 17: "lower_lip_outer",
    13: "upper_lip_inner", 14: "lower_lip_inner",
}
FACEMESH = {str(i): _FACEMESH_SEMANTIC.get(i, f"facemesh_{i}") for i in range(478)}

# Face contours (lips, eyes, eyebrows, face oval) plus both iris rings, as
# (start, end) landmark-id pairs. Copied from MediaPipe's FaceLandmarksConnections
# (Apache-2.0); used to draw FACEMESH keypoints.
FACEMESH_CONTOURS = (
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321),
    (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
    (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308), (78, 191), (191, 80),
    (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336),
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133),
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107),
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10), (474, 475), (475, 476), (476, 477), (477, 474),
    (469, 470), (470, 471), (471, 472), (472, 469),
)

COCO_WHOLEBODY_NAMES = {v: k for k, v in COCO_WHOLEBODY.items()}
COCO_NAMES = {v: k for k, v in COCO.items()}


HALPE_KEYPOINT_DICT = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
        
        17: "head_top",  # Calculate as middle of left_eye and right_eye 
        18: "neck",  # Calculate as center of nose, left_shoulder, right_shoulder
        19: "pelvis_point", # Calculate as middle of left_hip and right_hip

        20: "left_big_toe",    # 17
        21: "right_big_toe",   # 20
        22: "left_small_toe",  # 18
        23: "right_small_toe", # 21
        
        24: "left_heel",       # 19
        25: "right_heel"       # 22 (fixed typo: "heal" -> "heel")
    }
    
HALPE_TO_COCO_KEYPOINT_MAP = {
        0: 0,   # nose -> nose
        1: 1,   # left_eye -> left_eye
        2: 2,   # right_eye -> right_eye
        3: 3,   # left_ear -> left_ear
        4: 4,   # right_ear -> right_ear
        5: 5,   # left_shoulder -> left_shoulder
        6: 6,   # right_shoulder -> right_shoulder
        7: 7,   # left_elbow -> left_elbow
        8: 8,   # right_elbow -> right_elbow
        9: 9,   # left_wrist -> left_wrist
        10: 10, # right_wrist -> right_wrist
        11: 11, # left_hip -> left_hip
        12: 12, # right_hip -> right_hip
        13: 13, # left_knee -> left_knee
        14: 14, # right_knee -> right_knee
        15: 15, # left_ankle -> left_ankle
        16: 16, # right_ankle -> right_ankle
        17: None,  # head_top - calculated
        18: None,  # neck - calculated
        19: None,  # pelvis_point - calculated
        20: 17,    # left_big_toe
        21: 20,    # right_big_toe
        22: 18,    # left_small_toe
        23: 21,    # right_small_toe
        24: 19,    # left_heel
        25: 22,    # right_heel
    }

HUMAN26M = {
            0: 'root',
            1: 'right_hip',
            2: 'right_knee',
            3: 'right_ankle',
            4: 'left_hip',
            5: 'left_knee',
            6: 'left_ankle',
            7: 'torso',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'left_shoulder',
            12: 'left_elbow',
            13: 'left_wrist',
            14: 'right_shoulder',
            15: 'right_elbow',
            16: 'right_wrist',
            # Derived centroids appended by the motion helpers (add_*_centroid).
            133: 'head_centroid',
            134: 'body_centroid',
            135: 'pelvic_centroid'
        }


HUMAN26M_NAMES = {v: k for k, v in HUMAN26M.items()}
