"""Joint-angle overlay — display live anatomical angles on the side of the frame.

Two ways to use it:

1. End-to-end through the ``Video`` orchestrator (recommended) — set
   ``plot_angles=True``; the panel is rendered on the LEFT side automatically.

2. Standalone ``JointAnglePlotter`` — drive it yourself per frame and composite
   the transparent panel onto any image.

The angles measured are the eight major anatomical joints (left/right shoulder,
elbow, hip, knee), defined once in
``physiotrack.signals.motion.features.JOINT_ANGLE_TRIPLETS``.
"""

import cv2
import physiotrack as pt
from physiotrack.signals import JointAnglePlotter


# --- 1) End-to-end via the Video orchestrator --------------------------------
def run_pipeline(video_path: str):
    pt.Video(
        source=video_path,
        detector=pt.Detection.Person(),
        pose=pt.Pose.Person(),          # whole-body pose feeds the angle panel
        plot_angles=True,               # <- enables the left-side joint-angle overlay
        # angle_joints=["leftElbow", "rightElbow", "leftKnee", "rightKnee"],  # optional subset
    ).run(output_video="out_angles.mp4", output_json="poses.json")


# --- 2) Standalone, frame by frame -------------------------------------------
def run_standalone(video_path: str):
    pose = pt.Pose.Person()
    plotter = JointAnglePlotter(rom=True, fps=30.0)   # 8 joints + clinical ROM

    cap = cv2.VideoCapture(video_path)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = pose.predict(frame)             # -> Result
        pose_results = result.to_dict()["instances"]    # [{'keypoints': [...]}, ...]
        plotter.update(pose_results, frame_time=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
        # joint-angle grid + ROM grid (2-column L|R panels) stacked on the left
        frame = plotter.attach_to_frame(frame, position="top_left")
        cv2.imshow("joint angles", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    run_pipeline(src)
