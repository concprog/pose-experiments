import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import FastSAM


def load_voc_classes(root: str) -> list[str]:
    """Load standard VOC classes"""
    try:
        with open(os.path.join(root, "labels.txt"), "r") as f:
            names = f.read().splitlines()
            names = list(filter(len, names))
        return names
    except FileNotFoundError:
        return [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]


def extract_all_bboxes(
    voc_root: str, split: str
) -> dict[str, dict[str, list[tuple[int, int, int, int]]]]:
    """Extracts all bounding boxes organized by image and class"""
    annotations_dir = os.path.join(voc_root, "Annotations")
    imagesets_dir = os.path.join(voc_root, "ImageSets", "Main")

    split_file = os.path.join(imagesets_dir, f"{split}.txt")
    with open(split_file, "r") as f:
        image_ids = [line.strip() for line in f.readlines()]

    bbox_dict = {}

    for img_id in tqdm(image_ids, desc="Extracting bboxes"):
        xml_path = os.path.join(annotations_dir, f"{img_id}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        img_boxes = {}
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            bndbox = obj.find("bndbox")
            bbox = (
                float(bndbox.find("xmin").text),
                float(bndbox.find("ymin").text),
                float(bndbox.find("xmax").text),
                float(bndbox.find("ymax").text),
            )
            img_boxes.setdefault(class_name, []).append(bbox)

        if img_boxes:
            filename = root.find("filename").text
            bbox_dict[filename] = img_boxes

    return bbox_dict


def generate_voc_segmentation_with_fastsam(
    voc_root: str,
    split: str = "train",
    model_size: str = "s",  # 's' or 'x'
    imgsz: int = 1024,
    conf: float = 0.4,
    iou: float = 0.9,
    overwrite: bool = False,
):
    """
    Generates VOC-format segmentation masks using FastSAM

    Args:
        voc_root: Path to VOC2012 directory
        split: Dataset split ('train', 'val', 'trainval')
        model_size: Model size ('s' for small, 'x' for extra small)
        imgsz: Inference size (pixels)
        conf: Confidence threshold
        iou: IoU threshold
        overwrite: Whether to overwrite existing masks
    """
    # Path setup
    jpeg_dir = os.path.join(voc_root, "JPEGImages")
    seg_class_dir = os.path.join(voc_root, "SegmentationClass")
    seg_obj_dir = os.path.join(voc_root, "SegmentationObject")
    seg_class_color_dir = os.path.join(voc_root, "SegmentationClassColor")  # New directory for color masks

    os.makedirs(seg_class_dir, exist_ok=True)
    os.makedirs(seg_obj_dir, exist_ok=True)
    os.makedirs(seg_class_color_dir, exist_ok=True)  # Create color directory

    # Generate random color palette for 21 classes (0-20)
    np.random.seed(42)  # Fixed seed for consistent colors
    palette = np.random.randint(0, 256, (21, 3), dtype=np.uint8)

    # Load model and classes
    model = FastSAM(f"FastSAM-{model_size}.pt")
    classes = load_voc_classes(voc_root)

    # Get all bounding boxes
    bbox_dict = extract_all_bboxes(voc_root, split)

    for img_name, class_boxes in tqdm(bbox_dict.items(), desc="Generating masks"):
        base_name = os.path.splitext(img_name)[0]
        class_mask_path = os.path.join(seg_class_dir, f"{base_name}.png")
        obj_mask_path = os.path.join(seg_obj_dir, f"{base_name}.png")
        class_color_path = os.path.join(seg_class_color_dir, f"{base_name}.png")  # Color mask path

        # Skip if masks exist and not overwriting
        if (
            not overwrite
            and os.path.exists(class_mask_path)
            and os.path.exists(obj_mask_path)
            and os.path.exists(class_color_path)  # Also check color mask
        ):
            continue

        img_path = os.path.join(jpeg_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        class_mask = np.zeros((h, w), dtype=np.uint8)
        instance_mask = np.zeros((h, w), dtype=np.uint16)

        for class_name, bboxes in class_boxes.items():
            class_id = classes.index(class_name) + 1  # Background=0

            # Convert bboxes to FastSAM format (list of lists)
            bbox_list = [list(bbox) for bbox in bboxes]

            # Run FastSAM inference with bbox prompts
            results = model(
                img_path,
                bboxes=bbox_list,
                device="cuda" if torch.cuda.is_available() else "cpu",
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                retina_masks=True,
                verbose=False,
            )

            if results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()
                for i, mask in enumerate(masks):
                    # Resize mask to original image dimensions if needed
                    if mask.shape[0] != h or mask.shape[1] != w:
                        mask = cv2.resize(
                            mask.astype(np.uint8),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    # Update semantic segmentation mask
                    class_mask[mask > 0] = class_id

                    # Update instance segmentation mask (VOC format)
                    instance_mask[mask > 0] = class_id * 1000 + (i + 1)

        # Save original grayscale masks
        cv2.imwrite(class_mask_path, class_mask)
        cv2.imwrite(obj_mask_path, instance_mask)

        # Create and save colorized version
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in range(21):  # Process all 21 classes
            color_mask[class_mask == idx] = palette[idx]
        cv2.imwrite(class_color_path, color_mask)

if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(
        description="Generate VOC-format segmentation masks using FastSAM"
    )

    # Required arguments
    parser.add_argument(
        "--voc_root",
        type=str,
        required=True,
        help="Path to VOC2012 directory (e.g., '/datasets/data/VOCdevkit/VOC2012')",
    )

    # Optional arguments
    parser.add_argument(
        "--split",
        type=str,
        default="trainval",
        choices=["train", "val", "trainval"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="s",
        choices=["s", "x"],
        help="FastSAM model size ('s' for small, 'x' for extra small)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1024, help="Inference size in pixels"
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="Confidence threshold for detection"
    )
    parser.add_argument(
        "--iou", type=float, default=0.9, help="IoU threshold for mask generation"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing mask files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)",
    )

    args = parser.parse_args()

    generate_voc_segmentation_with_fastsam(
        voc_root=args.voc_root,
        split=args.split,
        model_size=args.model_size,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        overwrite=args.overwrite,
    )
