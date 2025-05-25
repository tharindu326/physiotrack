import time
import cv2
import numpy as np
import torch
import os
from .common import create_preprocessor
import torch.nn.functional as F
from .classes_and_palettes import SEGMENTATION_CLASSES


random = np.random.RandomState(11)
colors = random.randint(0, 255, (len(SEGMENTATION_CLASSES) - 1, 3))
colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  # Add background color
colors = colors[:, ::-1]

class SapiensSegmentation():
    def __init__(self,
                 model,
                 device,
                 dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        model_folder = os.path.join(os.path.dirname(__file__), '..', 'model_data')
        model_path = os.path.join(model_folder, model.value)
        self.model = torch.jit.load(model_path).eval().to(self.device).to(dtype)
        self.preprocessor = create_preprocessor(input_size=(1024, 768))  
        

    def __call__(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        # Model expects BGR, but we change to RGB here because the preprocessor will switch the channels also
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)

        with torch.inference_mode():
            results = self.model(tensor)
        segmentation_map = postprocess_segmentation(results, img.shape[:2])

        print(f"Segmentation inference took: {time.perf_counter() - start:.4f} seconds")
        return segmentation_map
    
    def inference(self, img):
        start = time.perf_counter()
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)
        with torch.inference_mode():
            results = self.model(tensor)
        segmentation_map = postprocess_segmentation(results, img.shape[:2])
        print(f"Sapiense segmentation inference took: {time.perf_counter() - start:.4f} seconds")
        return segmentation_map


def draw_segmentation_map(segmentation_map: np.ndarray) -> np.ndarray:
    h, w = segmentation_map.shape #single channel
    segmentation_img = np.zeros((h, w, 3), dtype=np.uint8) #3 channels

    for i, color in enumerate(colors):
        segmentation_img[segmentation_map == i] = color

    return segmentation_img

def postprocess_segmentation(results: torch.Tensor, img_shape) -> np.ndarray:
    result = results[0].cpu()
    # Upsample the result to the original image size
    logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)
    # Perform argmax to get the segmentation map
    segmentation_map = logits.argmax(dim=0, keepdim=True)
    # Covert to numpy array
    segmentation_map = segmentation_map.float().numpy().squeeze()
    return segmentation_map


if __name__ == "__main__":
    from enum import Enum
    images = ['samples/BV_S17_frame_1.jpg', 'samples/BV_S17_frame_2.jpg', 'samples/BV_S17_frame_3.jpg']
    class Segmentation:
        class Sapiens(Enum):
            B1_PT = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth"
            B2_TS = "sapiens_2b_goliath_best_goliath_mIoU_8179_epoch_181_torchscript.pt2"  # https://huggingface.co/camenduru/sapiens-body-part-segmentation/resolve/main/sapiens_2b_goliath_best_goliath_mIoU_8179_epoch_181_torchscript.pt2?download=true
            B1_TS = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"  # https://huggingface.co/facebook/sapiens-seg-1b-torchscript/resolve/main/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2?download=true
            B03_TS = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"  # https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript/resolve/main/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2?download=true
            B06_TS = "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"  # https://huggingface.co/facebook/sapiens-seg-0.6b-torchscript/resolve/main/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2?download=true
            B1_BF16 = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_bfloat16.pt2"  # https://huggingface.co/facebook/sapiens-seg-1b-bfloat16/resolve/main/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_bfloat16.pt2?download=true
            B06_BF16 = "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_bfloat16.pt2"  # https://huggingface.co/facebook/sapiens-seg-0.6b-bfloat16/resolve/main/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_bfloat16.pt2?download=true
            B03_BF16 = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_bfloat16.pt2"  # https://huggingface.co/facebook/sapiens-seg-0.3b-bfloat16/resolve/main/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_bfloat16.pt2?download=true
    
    output_path = 'test_path'
    model = Segmentation.Sapiens.B1_TS
    device = 'cpu'
    os.makedirs(output_path, exist_ok=True)
    estimator = SapiensSegmentation(model, device)
    for img_path in images:
        output_path = os.path.join(output_path, f"segemneted_{model.name}_{img_path.split('/')[-1]}")
        img = cv2.imread(img_path)
        start = time.perf_counter()
        segmentations = estimator(img)
        print(f"Time taken: {time.perf_counter() - start:.4f} seconds")

        segmentation_img = draw_segmentation_map(segmentations)
        combined = cv2.addWeighted(img, 0.5, segmentation_img, 0.5, 0)
        cv2.imwrite(output_path, combined)
        
        # # Convert depth_img from BGR to RGB for visualization and saving
        # segmented_img_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        # height, width, _ = segmented_img_rgb.shape
        # fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)

        # # Display and save the RGB depth image
        # plt.imshow(segmented_img_rgb)
        # plt.axis('off')  # Hide the axis

        # # Save the depth image as RGB
        # plt.savefig("segmented_image_rgb.png", bbox_inches='tight', pad_inches=0)
        # plt.show()
