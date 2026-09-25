"""Head orientation of every face in an image.

Detects faces with the VR-tuned face detector, estimates each face's yaw / pitch / roll
with the VR-tuned 6DRepNet360 checkpoint, prints the angles and saves the annotated
image (``Result.plot`` draws the head axes).

Usage:
    python face3d.py kinect_s1_v1_frame1.png
    python face3d.py photo.jpg --device cpu --output out.png
"""
import argparse

import physiotrack as pt
from physiotrack.core.predictor import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Input image.")
    parser.add_argument("--device", default="0", help="cpu, cuda or a CUDA index (default: 0).")
    parser.add_argument("--output", default="face_orientation_output.png",
                        help="Annotated output image (default: face_orientation_output.png).")
    args = parser.parse_args()

    image = load_image(args.image)
    faces = pt.VRFace(device=args.device)(image)
    faces = pt.FaceOrientation(model=pt.Models.Face.Orientation.VR, device=args.device)(image, faces)

    for index, face in enumerate(faces):
        o = face.orientation
        print(f"face {index}: yaw {o['yaw']:+.1f}  pitch {o['pitch']:+.1f}  roll {o['roll']:+.1f} deg")
    faces.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
