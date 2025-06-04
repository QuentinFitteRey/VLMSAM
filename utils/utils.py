from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<|image_pad|>"
DEFAULT_IM_START_TOKEN = "<|vision_start|>"
DEFAULT_IM_END_TOKEN = "<|vision_end|>"


DEFAULT_SEMATIC_SEG =  " Generate a mask for semantic segmentation. Focus on identifying the category as a whole; do not distinguish between individual instances within the same category."

DEFAULT_INSTANT_SEG =  " Generate instance segmentation masks. Each countable object within the same category should have a separate mask. Ignore uncountable categories for individual masking."

DEFAULT_SEM_CONV_SEG = " Provide a textual answer and generate a semantic segmentation mask. Focus on categorizing the entire class without separating individual instances."

DEFAULT_INST_CONV_SEG    = " Respond with text and provide instance segmentation mask predictions. Different instances within the same countable category should be identified with separate masks. A single mask suffices for uncountable categories."

DEFAULT_PURE_CONV = " Respond to the question using text only. Do not generate any segmentation masks."

DEFAULT_COT = " Answer the question with a step-by-step explanation."

SHORT_QUESTION_LIST = [
     "\n" + "Can you segment the {class_name} in this image?"+ DEFAULT_SEMATIC_SEG,
    "\n" + "Please segment the {class_name} in this image."+ DEFAULT_SEMATIC_SEG,
     "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask."+ DEFAULT_SEMATIC_SEG,
    "\n"
    + "What is {class_name} in this image? Please output segmentation mask."+ DEFAULT_SEMATIC_SEG,
]

LONG_QUESTION_LIST = [
 "\n" + "{sent} Please respond with segmentation mask."+ DEFAULT_SEMATIC_SEG,
 "\n" + "{sent} Please output segmentation mask."+ DEFAULT_SEMATIC_SEG,
]

INST_LONG_QUESTION_LIST = [
    "\n" + "{sent} Please respond with segmentation masks." + DEFAULT_INSTANT_SEG,
     "\n" + "{sent} Please output segmentation masks." + DEFAULT_INSTANT_SEG,
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why."+ DEFAULT_SEM_CONV_SEG,
    "Please output segmentation mask and explain the reason."+ DEFAULT_SEM_CONV_SEG,
    "Please output segmentation mask and give some explanation."+ DEFAULT_SEM_CONV_SEG,
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

MULTI_CLASS_QUESTION_LIST = [
     "\n" + "Can you segment {classes} separately in this image?" + DEFAULT_SEMATIC_SEG,
     "\n" + "Can you segment {classes} respectively in this image?" + DEFAULT_SEMATIC_SEG,
     "\n" + "Please segment {classes} separately in this image." + DEFAULT_SEMATIC_SEG,
     "\n" + "Please segment {classes} respectively in this image." + DEFAULT_SEMATIC_SEG,
]

INST_QUESTION_LIST = [
     "\n" + "Can you segment the {class_name} separately in this image?" + DEFAULT_INSTANT_SEG,
     "\n" + "Can you segment the {class_name} respectively in this image?" + DEFAULT_INSTANT_SEG,
     "\n" + "Please segment the {class_name} separately in this image." + DEFAULT_INSTANT_SEG,
     "\n" + "Please segment the {class_name} respectively in this image." + DEFAULT_INSTANT_SEG,
     "\n" + "What are the {class_name} in this image? Please output segmentation masks separately." + DEFAULT_INSTANT_SEG,
     "\n"+ "What are the {class_name} in this image? Please output segmentation masks respectively." + DEFAULT_INSTANT_SEG,
]

INST_ANSWER_LIST = [
    "They are {seg_tokens}.",
    "Sure, {seg_tokens}.",
    "Sure, they are {seg_tokens}.",
    "Sure, the segmentation results are {seg_tokens}.",
    "{seg_tokens}.",
]



class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3



import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import auc

class Instance_Metric:
    def __init__(self) -> None:
        pass

    def iou(self, pred, target):
        intersection = (pred & target).float().sum((1, 2))
        union = (pred | target).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou

    def hungarian_match(self, pred, target):
        iou_matrix = torch.zeros((len(pred), len(target)))
        for i, p in enumerate(pred):
            for j, t in enumerate(target):
                iou_matrix[i, j] = self.iou(p.unsqueeze(0), t.unsqueeze(0))

        pred_indices, target_indices = linear_sum_assignment(-iou_matrix.cpu().numpy())
        return pred_indices, target_indices

    def precision_recall(self, matched_pred, matched_target, unmatched_pred, unmatched_target, thresholds):
        iou_scores = self.iou(matched_pred, matched_target)
        precisions = []
        recalls = []

        num_false_positives = len(unmatched_pred)
        num_false_negatives = len(unmatched_target)

        for threshold in thresholds:
            true_positive = (iou_scores > threshold).float().sum()
            false_positive = num_false_positives + (iou_scores <= threshold).float().sum()
            false_negative = num_false_negatives + len(matched_target) - true_positive
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
            precisions.append(precision.cpu())
            recalls.append(recall.cpu())

        return precisions, recalls

    def class_agnostic_map(self, pred, target, thresholds=torch.linspace(0.5, 0.95, 10)):
        pred_indices, target_indices = self.hungarian_match(pred, target)
        matched_pred = [pred[i] for i in pred_indices]
        matched_target = [target[i] for i in target_indices]

        unmatched_pred = [p for i, p in enumerate(pred) if i not in pred_indices]
        unmatched_target = [t for i, t in enumerate(target) if i not in target_indices]

        precisions, recalls = self.precision_recall(torch.stack(matched_pred), torch.stack(matched_target), unmatched_pred, unmatched_target, thresholds)

        if len(precisions) > 1 and len(recalls) > 1:
            ap = auc(recalls, precisions)
        else:
            ap = 0.0 
        return ap
    def class_agnostic_ap50(self, pred, target):
        return self.class_agnostic_map(pred, target, thresholds=torch.tensor([0.5]))

    def class_agnostic_ap75(self, pred, target):
        ap75 = self.class_agnostic_map(pred, target, thresholds=torch.tensor([0.75]))
        return ap75

    def mean_iou(self, pred, target):
        pred_indices, target_indices = self.hungarian_match(pred, target)
        matched_pred = [pred[i] for i in pred_indices]
        matched_target = [target[i] for i in target_indices]

        iou_scores = self.iou(torch.stack(matched_pred), torch.stack(matched_target))
        return torch.mean(iou_scores)

    def mask_recall(self, pred, target, threshold=0.5):
        pred_indices, target_indices = self.hungarian_match(pred, target)
        matched_pred = [pred[i] for i in pred_indices]
        matched_target = [target[i] for i in target_indices]

        iou_scores = self.iou(torch.stack(matched_pred), torch.stack(matched_target))
        true_positive = (iou_scores > threshold).float().sum()
        total = true_positive + len(target) - len(matched_target)
        return true_positive / total if total > 0 else 0


class Instance_Metric_New:
    def __init__(self) -> None:
        self.all_precision_lst = []
        self.all_recall_lst = []
        pass

    def iou(self, pred, target):
        intersection = (pred & target).float().sum((1, 2))
        union = (pred | target).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou

    def hungarian_match(self, pred, target):
        iou_matrix = torch.zeros((len(pred), len(target)))
        for i, p in enumerate(pred):
            for j, t in enumerate(target):
                iou_matrix[i, j] = self.iou(p.unsqueeze(0), t.unsqueeze(0))

        pred_indices, target_indices = linear_sum_assignment(-iou_matrix.cpu().numpy())
        return pred_indices, target_indices

    def precision_recall(self, matched_pred, matched_target, unmatched_pred, unmatched_target, thresholds):
        iou_scores = self.iou(matched_pred, matched_target)
        precisions = []
        recalls = []

        num_false_positives = len(unmatched_pred)
        num_false_negatives = len(unmatched_target)

        for threshold in thresholds:
            true_positive = (iou_scores > threshold).float().sum()
            false_positive = num_false_positives + (iou_scores <= threshold).float().sum()
            false_negative = num_false_negatives + len(matched_target) - true_positive
            precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
            recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
            precisions.append(precision.cpu())
            recalls.append(recall.cpu())

        return precisions, recalls

    def calaculate_pre_recall_map(self, pred, target, thresholds=torch.linspace(0.5, 0.95, 10)):
        pred_indices, target_indices = self.hungarian_match(pred, target)
        matched_pred = [pred[i] for i in pred_indices]
        matched_target = [target[i] for i in target_indices]

        unmatched_pred = [p for i, p in enumerate(pred) if i not in pred_indices]
        unmatched_target = [t for i, t in enumerate(target) if i not in target_indices]

        precisions, recalls = self.precision_recall(torch.stack(matched_pred), torch.stack(matched_target), unmatched_pred, unmatched_target, thresholds)
        self.all_precision_lst.append(precisions)
        self.all_recall_lst.append(recalls)


    def class_agnostic_ap50(self):
        #index = 0
        tmp_percision = torch.stack([torch.tensor(lst) for lst in self.all_precision_lst])
        tmp_ap50_per = tmp_percision[:,0]
        tmp_recall = torch.stack([torch.tensor(lst) for lst in self.all_recall_lst])
        tmp_ap50_rec = tmp_recall[:,0]
        sorted_indices = torch.argsort(tmp_ap50_rec)
        sorted_rec = tmp_ap50_rec[sorted_indices]
        sorted_per = tmp_ap50_per[sorted_indices]
        ap50 = auc(sorted_rec, sorted_per)
        return ap50

    def class_agnostic_ap75(self):
        #index = 5
        tmp_percision = torch.stack([torch.tensor(lst) for lst in self.all_precision_lst])
        tmp_ap75_per = tmp_percision[:,5]
        tmp_recall = torch.stack([torch.tensor(lst) for lst in self.all_recall_lst])
        tmp_ap75_rec = tmp_recall[:,5]
        
        sorted_indices = torch.argsort(tmp_ap75_rec)
        sorted_rec = tmp_ap75_rec[sorted_indices]
        sorted_per = tmp_ap75_per[sorted_indices]
        ap75 = auc(sorted_rec, sorted_per)
        return ap75

    def class_agnostic_map(self):
        #index = 5
        tmp_percision = torch.stack([torch.tensor(lst) for lst in self.all_precision_lst])
        tmp_recall = torch.stack([torch.tensor(lst) for lst in self.all_recall_lst])
        All_ap = []
        for idx in range(len(tmp_percision[0])):
            tmp_ap75_rec = tmp_recall[:,idx]
            tmp_ap75_per = tmp_percision[:,idx]

            
            sorted_indices = torch.argsort(tmp_ap75_rec)
            sorted_rec = tmp_ap75_rec[sorted_indices]
            sorted_per = tmp_ap75_per[sorted_indices]
            ap75 = auc(sorted_rec, sorted_per)
            All_ap.append(ap75)

        return sum(All_ap)/len(All_ap)

    
    def mean_iou(self, pred, target):
        
        pred_indices, target_indices = self.hungarian_match(pred, target)
        matched_pred = [pred[i] for i in pred_indices]
        matched_target = [target[i] for i in target_indices]

        iou_scores = self.iou(torch.stack(matched_pred), torch.stack(matched_target))
        return torch.mean(iou_scores)

    def mask_recall(self, pred, target, threshold=0.5):
        
        pred_indices, target_indices = self.hungarian_match(pred, target)
        matched_pred = [pred[i] for i in pred_indices]
        matched_target = [target[i] for i in target_indices]

        iou_scores = self.iou(torch.stack(matched_pred), torch.stack(matched_target))
        true_positive = (iou_scores > threshold).float().sum()
        total = true_positive + len(target) - len(matched_target)
        return true_positive / total if total > 0 else 0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict

def dict_to_cuda2(input_dict):
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif isinstance(v, dict):
            input_dict[k] = dict_to_cuda(v)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict




class CoCo_Save:
    def __init__(self) -> None:
        self.coco_detections = []
        self.coco_annotations = {
            "images": [],
            "categories": [{"id": 1, "name": "category1"}],
            "annotations": []
        }
        self.ann_id = 1
        self.img_id = 1

    
    def add_data(self, pred_masks, gt_masks, pred_scores):
        
        pred_masks = pred_masks.cpu().numpy()
        gt_masks = gt_masks.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        self.coco_annotations["images"].append({"id": self.img_id, "width": pred_masks.shape[2], "height": pred_masks.shape[1]})

        for mask in gt_masks:
            mask = mask.astype('uint8')
            rle = mask_utils.encode(np.asfortranarray(mask))  # 转换为RLE格式
            rle['counts'] = rle['counts'].decode('utf-8')
            
            area = mask_utils.area(rle)
            bbox = mask_utils.toBbox(rle).tolist()
            self.coco_annotations["annotations"].append({
                "id": self.ann_id,
                "image_id": self.img_id,
                "category_id": 1,
                "segmentation": rle,
                "area": int(area),
                "iscrowd": 0,
                "bbox": bbox  # Example bbox
            })
            self.ann_id += 1

        for pred_idx in range(len(pred_masks)):
            mask = pred_masks[pred_idx]
            mask = mask.astype('uint8')
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            area = mask_utils.area(rle)
            bbox = mask_utils.toBbox(rle).tolist()
            self.coco_detections.append({
                "image_id": self.img_id,
                "category_id": 1,
                "segmentation": rle,
                "area": int(area),
                "score": float(pred_scores[pred_idx])  # Example score
            })

        self.img_id += 1


class CoCo_Save_Multi:
    def __init__(self, rank, world_size) -> None:
        self.rank = rank
        self.world_size = world_size
        self.coco_detections = []
        self.coco_annotations = {
            "images": [],
            "categories": [{"id": 1, "name": "category1"}],
            "annotations": []
        }
        # Assign unique starting IDs based on rank
        self.ann_id = rank + 1
        self.img_id = rank + 1
        # Increment ids by world size to ensure no overlap
        self.ann_id_step = world_size
        self.img_id_step = world_size

    def add_data(self, pred_masks, gt_masks, pred_scores):
        # Convert tensor masks to numpy and process each mask
        pred_masks = pred_masks.cpu().numpy()
        gt_masks = gt_masks.cpu().numpy()
        pred_scores = pred_scores.cpu().numpy()
        
        # Add images entry
        self.coco_annotations["images"].append({"id": self.img_id, "width": pred_masks.shape[2], "height": pred_masks.shape[1]})
        
        # Add ground truth masks to annotations
        for mask in gt_masks:
            mask = mask.astype('uint8')
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            area = mask_utils.area(rle)
            bbox = mask_utils.toBbox(rle).tolist()
            
            self.coco_annotations["annotations"].append({
                "id": self.ann_id,
                "image_id": self.img_id,
                "category_id": 1,
                "segmentation": rle,
                "area": int(area),
                "iscrowd": 0,
                "bbox": bbox
            })
            self.ann_id += self.ann_id_step
        
        # Add predicted masks to detections
        for idx, mask in enumerate(pred_masks):
            mask = mask.astype('uint8')
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('utf-8')
            area = mask_utils.area(rle)
            bbox = mask_utils.toBbox(rle).tolist()
            
            self.coco_detections.append({
                "image_id": self.img_id,
                "category_id": 1,
                "segmentation": rle,
                "area": int(area),
                "score": float(pred_scores[idx])
            })
        
        # Increment the image ID by world size
        self.img_id += self.img_id_step


    def gather_all_data(self):
        """
        Gathers annotations and detections from all processes in the distributed training group.
        This method should be called by all ranks, but only rank 0 will process and combine the data.
        """
        # Prepare containers for the data to be gathered
        all_annotations = [None] * self.world_size
        all_detections = [None] * self.world_size
        
        # Log before starting the gather operation
        logging.info(f"Rank {dist.get_rank()}: Starting to gather all data.")
        
        # Ensure all processes participate in gathering
        try:
            dist.all_gather_object(all_annotations, self.coco_annotations)
            dist.all_gather_object(all_detections, self.coco_detections)
        except Exception as e:
            logging.error(f"Rank {dist.get_rank()}: Failed during gather operation with error: {e}")
            raise
        
        # Log after data has been gathered
        logging.info(f"Rank {dist.get_rank()}: Successfully gathered all data.")

        if dist.get_rank() == 0:
            # Process and combine the data only on rank 0
            combined_annotations = {
                "images": [],
                "categories": [{"id": 1, "name": "category1"}],
                "annotations": []
            }
            combined_detections = []
            
            # Combine the data gathered from all ranks
            for ann in all_annotations:
                combined_annotations["images"].extend(ann["images"])
                combined_annotations["annotations"].extend(ann["annotations"])
            
            for det in all_detections:
                combined_detections.extend(det)
            
            # Update local data with the combined results
            self.coco_annotations = combined_annotations
            self.coco_detections = combined_detections
            
            # Validate and log the final combined data
            logging.info("Rank 0: Data combined and processed.")