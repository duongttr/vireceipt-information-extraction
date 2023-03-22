import numpy as np
from dataset import ner_tags_list
from datasets import load_metric
from PIL import ImageDraw, ImageFont
import json
import pandas as pd


metric = load_metric("seqeval")


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000)
    ]


def normalize_box(bbox, width, height):
    return [
        int((bbox[0] / width) * 1000),
        int((bbox[1] / height) * 1000),
        int((bbox[2] / width) * 1000),
        int((bbox[3] / height) * 1000)
    ]

def draw_output(image, true_predictions, true_boxes):
    def iob_to_label(label):
        label = label
        if not label:
            return 'other'
        return label
    
    # width, height = image.size
    
    # predictions = logits.argmax(-1).squeeze().tolist()
    # is_subword = np.array(offset_mapping)[:,0] != 0
    # true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    # true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
    
    
    
    # draw
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline='red')
        draw.text((box[0] + 10, box[1] - 10),
                  text=predicted_label, fill='red', font=font)

    return image

def ReFormatter(text, cls):
    return text

def compute_metrics(p, return_entity_level_metrics=False):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [ner_tags_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ner_tags_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions, references=true_labels)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


# def create_json(boxes, texts, labels,
#                 chosen_labels=['ADDR', 'BILLID', 'DATETIME', 'CASHIER', 'SHOP_NAME',
#                                'PHONE', 'NUMBER', 'PRODUCT_NAME', 'UNIT', 'AMOUNT',
#                                'UPRICE', 'TAMOUNT', 'TPRICE', 'SUB_TPRICE']):


#     return json.dumps(final_result)
