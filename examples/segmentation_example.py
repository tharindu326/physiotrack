
from physiotrack import Segmentation, Models
import cv2
from pathlib import Path

def run_segmentation_example():
    """
    Run segmentation on a sample image

    This example shows how to use the Segmentation class similar to
    how Detection.Person() and Pose are used in the library.
    """

    # ===================================================================
    # Initialize Segmentors (similar to how Detection.Person is initialized)
    # ===================================================================
    print("Initializing segmentors...")

    # Option 1: Using YOLO VRHEAD preset for VR head/face/neck segmentation
    # This is similar to Detection.Person() or Detection.VRStudent()
    segmentor_vrhead = Segmentation.VRHEAD(
        device='cpu',
        OBJECTNESS_CONFIDENCE=0.24,
        NMS_THRESHOLD=0.4,
        classes=[0, 1, 2],  # VR head, face, neck classes
        render_segmenttion_map=True,
        verbose=False
    )

    # Option 2: Using YOLO Person preset for general person segmentation
    segmentor_person = Segmentation.Person(
        device='cpu',
        OBJECTNESS_CONFIDENCE=0.24,
        NMS_THRESHOLD=0.4,
        render_segmenttion_map=True,
        verbose=False
    )

    # Option 3: Using Custom with explicit model (like Pose.Custom())
    segmentor_custom = Segmentation.Custom(
        model=Models.Segmentation.Yolo.VRHEAD.M11,
        device='cpu',
        OBJECTNESS_CONFIDENCE=0.24,
        NMS_THRESHOLD=0.4,
        classes=[0, 1, 2],
        render_segmenttion_map=True,
        verbose=False
    )

    # Option 4: Using Sapiens for body part segmentation (27 body parts)
    # Note: This requires downloading a large model (~1.36GB for B03, ~3GB for B1)
    # Uncomment to use:
    # segmentor_bodypart = Segmentation.BodyPart(
    #     device='cpu',  # or 'cuda' for GPU
    #     render_segmenttion_map=True,
    #     verbose=False
    # )

    print("✓ All segmentors initialized successfully!")

    # ===================================================================
    # Example Usage with an Image
    # ===================================================================
    # Uncomment the following code if you have a test image

    # input_image_path = 'samples/test_image.jpg'
    # if Path(input_image_path).exists():
    #     print(f"\nProcessing image: {input_image_path}")
    #     frame = cv2.imread(input_image_path)
    #
    #     # Perform segmentation (similar to detector.detect() or pose.estimate())
    #     segmentation_img, segmentation_map = segmentor_vrhead.segment(frame)
    #
    #     # Overlay the segmentation on the original image
    #     output_image = cv2.addWeighted(frame, 0.5, segmentation_img, 0.5, 0)
    #
    #     # Save the result
    #     output_path = 'output/segmented_output.jpg'
    #     Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    #     cv2.imwrite(output_path, output_image)
    #     print(f"✓ Segmentation complete! Saved to: {output_path}")
    #
    #     # The segmentation_map is a 2D numpy array where each pixel value
    #     # represents the class ID of the segmented region
    #     print(f"Segmentation map shape: {segmentation_map.shape}")
    #     print(f"Unique classes found: {np.unique(segmentation_map)}")
    # else:
    #     print(f"Note: Sample image not found at {input_image_path}")

    # ===================================================================
    # Integration with Video Processing Pipeline
    # ===================================================================
    print("\n" + "="*60)
    print("Integration Example (like in motion.py):")
    print("="*60)
    print("""
    from physiotrack import Segmentation, Models

    # Initialize segmentor (similar to detector)
    segmentor = Segmentation.VRHEAD(
        device='cuda',
        OBJECTNESS_CONFIDENCE=0.24,
        NMS_THRESHOLD=0.4,
        classes=[0, 1, 2],
        render_segmenttion_map=True,
        verbose=False
    )

    # Or use Sapiens for body parts
    segmentor = Segmentation.BodyPart(
        device='cuda',
        render_segmenttion_map=True
    )

    # Or use Custom with any model
    segmentor = Segmentation.Custom(
        model=Models.Segmentation.Yolo.VRHEAD.M11,
        device='cuda'
    )

    # Use in video processing loop
    cap = cv2.VideoCapture('video.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform segmentation
        seg_img, seg_map = segmentor.segment(frame)

        # Process segmentation results...
        # seg_map contains class IDs for each pixel
        # seg_img contains the colored segmentation overlay
    """)
    print("="*60)

    print("\n=== Segmentation Example Complete ===")
    print("\nAvailable Segmentation Options:")
    print("  1. Segmentation.VRHEAD()     - YOLO VR head/face/neck segmentation")
    print("  2. Segmentation.Person()     - YOLO person segmentation")
    print("  3. Segmentation.BodyPart()   - Sapiens body part segmentation (27 parts)")
    print("  4. Segmentation.Custom()     - Custom model selection")
    print("\nAll segmentation classes follow the same pattern as Detection and Pose!")

if __name__ == "__main__":
    run_segmentation_example()
