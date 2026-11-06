"""Face-part parsing with SegFace, and the facial regions of each face.

``Segmentation.Face`` parses faces into 19 CelebAMask-HQ classes (skin, eyes, brows,
nose, lips, hair, ears, glasses, hat, earring, necklace, neck, cloth, ...) and draws the
colour overlay. The ``FaceRegions`` face stage runs the same parser per face and reports
each face's regions -- the share of the face box per class, e.g. how much of it is
visible skin (the rPPG signal source). Faces come from ``pt.Face``, or with ``--vr``
from the VR-tuned ``pt.VRFace``.

Usage:
    python face_parsing.py kinect_s1_v1_frame1.png
    python face_parsing.py photo.jpg --vr --device cpu
"""
import argparse

import physiotrack as pt
from physiotrack.core.predictor import load_image


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Input image.")
    parser.add_argument("--vr", action="store_true",
                        help="Find faces with the VR-tuned detector (headset wearers).")
    parser.add_argument("--device", default="0", help="cpu, cuda or a CUDA index (default: 0).")
    parser.add_argument("--output", default="face_parsing_output.png",
                        help="Annotated output image (default: face_parsing_output.png).")
    args = parser.parse_args()

    image = load_image(args.image)
    faces = (pt.VRFace if args.vr else pt.Face)(device=args.device)(image)

    for index, face in enumerate(pt.FaceRegions(device=args.device)(image, faces)):
        shares = face.regions["fractions"]
        top = ", ".join(f"{name} {share:.0%}" for name, share in
                        sorted(shares.items(), key=lambda kv: -kv[1])[:5])
        print(f"face {index} {face.box.round().astype(int).tolist()}: {top}")

    pt.Segmentation.Face(device=args.device).predict(image, boxes=faces.boxes).save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
