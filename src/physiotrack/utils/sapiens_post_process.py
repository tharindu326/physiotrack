import numpy as np
import cv2
from physiotrack.core import cfg


def filter_by_box(segmentation_map, bbox=None):
    filtered_map = np.zeros_like(segmentation_map, dtype=segmentation_map.dtype)
    if bbox is None:
        return segmentation_map
    x1, y1, x2, y2 = bbox[0]
    unique_classes = np.unique(segmentation_map)
    for cls in unique_classes:
        if cls == 0:
            continue
        mask = np.uint8(segmentation_map == cls) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            is_within_bbox = all((x1 <= point[0][0] <= x2) and (y1 <= point[0][1] <= y2) for point in contour)
            if is_within_bbox:
                cv2.drawContours(filtered_map, [contour], -1, int(cls), thickness=cv2.FILLED)
    return filtered_map

def exclude_contours(segmentation_map, exclude):
    for cls in exclude:
        mask = np.uint8(segmentation_map == cls) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(segmentation_map, [contour], -1, 0, thickness=cv2.FILLED)
    return segmentation_map

def combine_contours(segmentation_map, combine):
    for src_class, target_class in combine.items():
        src_class = int(src_class)
        target_class = int(target_class)
        combined_mask = np.zeros_like(segmentation_map, dtype=np.uint8)
        mask = np.uint8(segmentation_map == src_class) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(combined_mask, [contour], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(segmentation_map, [contour], -1, 0, thickness=cv2.FILLED)
        combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in combined_contours:
            cv2.drawContours(segmentation_map, [contour], -1, target_class, thickness=cv2.FILLED)
    return segmentation_map


def remove_isolated_contours(segmentation_map):
    binary_map = np.uint8(segmentation_map > 0) * 255
    dilated_map = cv2.dilate(binary_map, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(segmentation_map)
    # get the large connected contours
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_mask = np.zeros_like(binary_map)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    # Use element-wise multiplication to retain only the largest connected component
    filtered_map = segmentation_map * (largest_contour_mask // 255)
    
    return filtered_map

def filter_contours_by_area(segmentation_map, min_area=0, max_area=None, exclude=[]):
    filtered_map = np.zeros_like(segmentation_map, dtype=segmentation_map.dtype)
    unique_classes = np.unique(segmentation_map)

    for cls in unique_classes:
        if cls == 0:
            continue
        mask = np.uint8(segmentation_map == cls) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cls in exclude:
                cv2.drawContours(filtered_map, [contour], -1, int(cls), thickness=cv2.FILLED)
            else:
                # if the class is not subject to area filtering, do the area filtering 
                area = cv2.contourArea(contour)
                if area >= min_area and (max_area is None or area <= max_area):
                    cv2.drawContours(filtered_map, [contour], -1, int(cls), thickness=cv2.FILLED)
    return filtered_map


def filter_by_connectivity(segmentation_map, connectivity_threshold=50):
    """
    Filter out components that do not meet the minimum connectivity threshold.
    """
    filtered_map = np.zeros_like(segmentation_map, dtype=segmentation_map.dtype)
    unique_classes = np.unique(segmentation_map)

    for cls in unique_classes:
        if cls == 0:
            continue
        mask = np.uint8(segmentation_map == cls)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        for i in range(1, num_labels):  # Start from 1 to skip the background
            connected_pixel_count = stats[i, cv2.CC_STAT_AREA]
            if connected_pixel_count >= connectivity_threshold:
                # Keep this connected component
                component_mask = (labels == i).astype(np.uint8) * 255
                filtered_map[component_mask == 255] = cls

    return filtered_map

def process_segmentation_map(segmentation_map, exclude=None, combine=None, min_area=0, max_area=None, exclude_in_area_filtering=[]):
    # Filter contours by area without the excluding classes
    if min_area > 0 or max_area is not None:
        segmentation_map = filter_contours_by_area(segmentation_map, min_area, max_area, exclude=exclude_in_area_filtering)
    
    # remove the unconnected components
    if cfg.sam_post_process.remove_unconnected:
        segmentation_map = filter_by_connectivity(segmentation_map)
    
    # remove isolated contours
    if cfg.sam_post_process.remove_isolated:
        segmentation_map = remove_isolated_contours(segmentation_map)
        
    # Exclude specified contours
    if exclude:
        segmentation_map = exclude_contours(segmentation_map, exclude)
    
    # Combine specified contours
    if combine:
        segmentation_map = combine_contours(segmentation_map, combine)
    
    return segmentation_map


def get_class_contours(segmentation_map, bbox=None):
    black_canvas = np.zeros_like(segmentation_map, dtype=np.uint8)
    black_canvas = cv2.cvtColor(black_canvas, cv2.COLOR_GRAY2BGR)
    
    unique_classes = np.unique(segmentation_map)
    
    # Draw individual contours
    for cls in unique_classes:
        if cls == 0:
            continue
        mask = np.uint8(segmentation_map == cls) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for idx, contour in enumerate(contours):
            label = f'{int(cls)}:{idx}'
            cv2.drawContours(black_canvas, [contour], -1, (0, 255, 0), 1)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(black_canvas, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if bbox is not None:     
        x1, y1, x2, y2 = bbox[0]
        cv2.rectangle(black_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return black_canvas

if __name__ == '__main__':
    im_path = 'samples/frame_1.png'
    frame = cv2.imread(im_path)
    segmentation_map = np.load("segmentation_map_full.npy")
    bbox = np.array([[421,  101,  647, 845]])
    
    segmentation_map = filter_by_box(segmentation_map, bbox=bbox)
    
    segmentation_map = process_segmentation_map(segmentation_map, 
                                                exclude=cfg.sam_post_process.exclude, 
                                                combine=cfg.sam_post_process.combine, 
                                                min_area=cfg.sam_post_process.min_area, 
                                                max_area=cfg.sam_post_process.max_area, 
                                                exclude_in_area_filtering=cfg.sam_post_process.exclude_in_area_filtering)
    
    np.save('segmentation_map_post_processed.npy', segmentation_map)
    black_canvas = get_class_contours(segmentation_map, bbox=None)
    output_path = "results/contours_black_canvas_function2.jpg"
    cv2.imwrite(output_path, black_canvas)
