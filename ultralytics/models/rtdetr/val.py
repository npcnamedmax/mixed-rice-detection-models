# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops
import numpy as np
__all__ = ("RTDETRValidator",)  # tuple or list


class RTDETRDataset(YOLODataset):
    """Real-Time DEtection and TRacking (RT-DETR) dataset class extending the base YOLODataset class.

    This specialized dataset class is designed for use with the RT-DETR object detection model and is optimized for
    real-time detection and tracking tasks.

    Attributes:
        augment (bool): Whether to apply data augmentation.
        rect (bool): Whether to use rectangular training.
        use_segments (bool): Whether to use segmentation masks.
        use_keypoints (bool): Whether to use keypoint annotations.
        imgsz (int): Target image size for training.

    Methods:
        load_image: Load one image from dataset index.
        build_transforms: Build transformation pipeline for the dataset.

    Examples:
        Initialize an RT-DETR dataset
        >>> dataset = RTDETRDataset(img_path="path/to/images", imgsz=640)
        >>> image, hw0, hw = dataset.load_image(0)
    """

    def __init__(self, *args, data=None, **kwargs):
        """Initialize the RTDETRDataset class by inheriting from the YOLODataset class.

        This constructor sets up a dataset specifically optimized for the RT-DETR (Real-Time DEtection and TRacking)
        model, building upon the base YOLODataset functionality.

        Args:
            *args (Any): Variable length argument list passed to the parent YOLODataset class.
            data (dict | None): Dictionary containing dataset information. If None, default values will be used.
            **kwargs (Any): Additional keyword arguments passed to the parent YOLODataset class.
        """
        super().__init__(*args, data=data, **kwargs)

        self.train_mode = "train" in self.prefix

        a=2.0
        t=0.0001
        self.count_instances()
        f_i_c = self.image_counts    / len(self.labels)              # class c / total images
        f_b_c = self.instance_counts / np.sum(self.instance_counts)  # class c / total bboxes
        class_weights = np.exp(a*(np.sqrt(t/(np.sqrt(
            f_i_c*f_b_c
        )))))
        

        # Aggregation function
        self.agg_func = np.max

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
    
    def count_instances(self):
        num_classes = len(self.data["names"])
        self.image_counts    = np.zeros(num_classes)  # for f_i_c: images containing class c
        self.instance_counts = np.zeros(num_classes)  # for f_b_c: total bboxes of class c

        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            seen = set()
            for id in cls:
                self.instance_counts[id] += 1
                if id not in seen:
                    self.image_counts[id] += 1  # count image once per class
                    seen.add(id)

        # Avoid division by zero
        self.image_counts    = np.where(self.image_counts    == 0, 1, self.image_counts)
        self.instance_counts = np.where(self.instance_counts == 0, 1, self.instance_counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
              weights.append(1)
              continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def load_image(self, i, rect_mode=False):
        """Load one image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool, optional): Whether to use rectangular mode for batch inference.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

        Examples:
            Load an image from the dataset
            >>> dataset = RTDETRDataset(img_path="path/to/images")
            >>> image, hw0, hw = dataset.load_image(0)
        """
        return super().load_image(i=i, rect_mode=rect_mode)

    def build_transforms(self, hyp=None):
        """Build transformation pipeline for the dataset.

        Args:
            hyp (dict, optional): Hyperparameters for transformations.

        Returns:
            (Compose): Composition of transformation functions.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, stretch=True)
        else:
            # transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), auto=False, scale_fill=True)])
            transforms = Compose([])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transforms
    
    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))


class RTDETRValidator(DetectionValidator):
    """RTDETRValidator extends the DetectionValidator class to provide validation capabilities specifically tailored for
    the RT-DETR (Real-Time DETR) object detection model.

    The class allows building of an RTDETR-specific dataset for validation, applies Non-maximum suppression for
    post-processing, and updates evaluation metrics accordingly.

    Attributes:
        args (Namespace): Configuration arguments for validation.
        data (dict): Dataset configuration dictionary.

    Methods:
        build_dataset: Build an RTDETR Dataset for validation.
        postprocess: Apply Non-maximum suppression to prediction outputs.

    Examples:
        Initialize and run RT-DETR validation
        >>> from ultralytics.models.rtdetr import RTDETRValidator
        >>> args = dict(model="rtdetr-l.pt", data="coco8.yaml")
        >>> validator = RTDETRValidator(args=args)
        >>> validator()

    Notes:
        For further details on the attributes and methods, refer to the parent DetectionValidator class.
    """

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build an RTDETR Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str, optional): `train` mode or `val` mode, users are able to customize different augmentations for
                each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (RTDETRDataset): Dataset configured for RT-DETR validation.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=False,  # no augmentation
            hyp=self.args,
            rect=False,  # no rect
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data,
        )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return predn

    def postprocess(
        self, preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        
        if not isinstance(preds, (list, tuple)):
            preds = [preds, None]

        bs, _, nd = preds[0].shape
        nc = self.nc

        # --- DEBUG SECTION START ---
        print(f"\n[DEBUG] RT-DETR Postprocess Check")
        print(f"  > Input Shape: {preds[0].shape}")
        print(f"  > Batch Size (bs): {bs}, Total Dim (nd): {nd}, Num Classes (nc): {nc}")
        
        expected_dim = 4 + nc + nc  # 4 Box + nc Score + nc Mass
        if nd != expected_dim:
            print(f"  > ⚠️ WARNING: Dimension Mismatch! Expected {expected_dim} but got {nd}.")
            print(f"  > This suggests the model export/forward pass structure does not match the validator.")
        else:
            print(f"  > Dimensions look correct for (Box + Score + Mass).")
        # --- DEBUG SECTION END ---

        # The Split Logic
        # CAUTION: If your model outputs [BBox, Mass, Score], this line will assign Mass data to 'scores' variable
        split_sizes = (4, nd - 4 - nc, nc)
        bboxes, scores, masses = preds[0].split(split_sizes, dim=-1)

        # --- DEBUG SECTION START ---
        print(f"  > Split Sizes Used: {split_sizes}")
        print(f"  > 'Scores' Stats -> Min: {scores.min().item():.4f}, Max: {scores.max().item():.4f}, Mean: {scores.mean().item():.4f}")
        print(f"  > 'Masses' Stats -> Min: {masses.min().item():.4f}, Max: {masses.max().item():.4f}, Mean: {masses.mean().item():.4f}")

        if scores.max().item() > 1.0:
            print(f"  > 🚨 CRITICAL: 'Scores' contains values > 1.0. This is suspicious.")
            print(f"    Possible Cause 1: Split order is wrong (Variable 'scores' actually holds Mass).")
            print(f"    Possible Cause 2: Model output is Logits, not Sigmoid.")
        # --- DEBUG SECTION END ---

        masses = masses.relu() # Ensure positivity
        
        # Standard Postprocessing
        bboxes *= self.args.imgsz
        outputs = [torch.zeros((0, 7), device=bboxes.device)] * bs
        
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1)  # (300, )
            
            selected_mass = torch.gather(masses[i], dim=-1, index=cls.unsqueeze(-1)).squeeze(-1)
            
            # Filter
            pred = torch.cat([bbox, score[..., None], cls[..., None], selected_mass[..., None]], dim=-1)
            
            # Sort by confidence
            pred = pred[score.argsort(descending=True)]
            
            # Apply Threshold
            filtered_pred = pred[score > self.args.conf]
            outputs[i] = filtered_pred

            # --- DEBUG SECTION (First Image Only) ---
            if i == 0:
                 print(f"  > Image 0 Processing:")
                 print(f"    - Raw Candidates: {pred.shape[0]}")
                 print(f"    - Confidence Threshold: {self.args.conf}")
                 print(f"    - Surviving Candidates: {filtered_pred.shape[0]}")
                 if filtered_pred.shape[0] > 0:
                     top = filtered_pred[0]
                     print(f"    - Top Det: Cls={int(top[5])}, Conf={top[4]:.4f}, Mass={top[6]:.4f}")
            # --- DEBUG SECTION END ---

        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "mass": x[:, 6]} for x in outputs]

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
                bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = predn["bboxes"].clone()
        box[..., [0, 2]] *= pbatch["ori_shape"][1] / self.args.imgsz  # native-space pred
        box[..., [1, 3]] *= pbatch["ori_shape"][0] / self.args.imgsz  # native-space pred
        box = ops.xyxy2xywh(box)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c, m in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist(), predn["mass"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                    "mass": round(m, 4)
                }
            )
