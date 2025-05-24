import os
from typing import List
import requests
from torchvision import transforms
from tqdm import tqdm
from . import get_model_category

def download_model(enum_item, download_path):
    os.makedirs(download_path, exist_ok=True)
    task = get_model_category(enum_item).lower()
    if task == 'segmentation':
        task = "seg"
        
    file_name = enum_item.value
    size = file_name.split('_')[1]
    if "03" in size:
        size = size.replace("03", "0.3")
    if "06" in size:
        size = size.replace("06", "0.6")
        
    format_type = file_name.split('_')[-1].split('.')[0]
    base_url = f"https://huggingface.co/facebook/sapiens-{task}-{size}-{format_type}/resolve/main"
    if "2" in size and task == "seg":
        base_url = f"https://huggingface.co/camenduru/sapiens-body-part-segmentation/resolve/main"
    
    download_url = f"{base_url}/{file_name}?download=true"
    print(download_url)
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(download_path, file_name)
        # Get total file size from headers
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        # Set up the progress bar
        tqdm_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=file_name)
        with open(file_path, 'wb') as f:
            for data in response.iter_content(block_size):
                tqdm_bar.update(len(data))
                f.write(data)
        tqdm_bar.close()
        if total_size != 0 and tqdm_bar.n != total_size:
            print("Error: Download may be incomplete.")
        else:
            print(f"Downloaded {file_name} to {download_path} successfully.")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")

def create_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               transforms.Lambda(lambda x: x.unsqueeze(0))
                               ])
