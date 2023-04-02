import json
import numpy as np
from collections import Counter
from datasets import load_metric
from PIL import ImageDraw, ImageFont

from src.data.dataset import ner_tags_list

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


def create_json(true_texts,
                true_predictions,
                true_boxes,
                chosen_labels=['SHOP_NAME', 'ADDR', 'TITLE', 'PHONE',
                               'PRODUCT_NAME', 'AMOUNT', 'UNIT', 'UPRICE', 'SUB_TPRICE', 'UDISCOUNT',
                               'TAMOUNT', 'TPRICE', 'FPRICE', 'TDISCOUNT',
                               'RECEMONEY', 'REMAMONEY',
                               'BILLID', 'DATETIME', 'CASHIER']
                ):

    result = {'PRODUCTS': {}}
    # products = []
    for text, prediction, box in zip(true_texts, true_predictions, true_boxes):
        if prediction not in chosen_labels:
            continue

        if prediction in ['AMOUNT', 'UNIT', 'UDISCOUNT', 'UPRICE', 'SUB_TPRICE',
                          'UDISCOUNT', 'TAMOUNT', 'TPRICE', 'FPRICE', 'TDISCOUNT',
                          'RECEMONEY', 'REMAMONEY']:
            text = reformat(text)

        if prediction in ['PRODUCT_NAME', 'AMOUNT', 'UNIT', 'UPRICE', 'SUB_TPRICE', 'UDISCOUNT']:
            # tlx,tly,brx,bry = box[0], box[1], box[2], box[3]
            # center_x, center_y = int((tlx+brx)/2), int((tly+bry)/2)
            # products.append({prediction: (text, (center_x, center_y))})
            if prediction in result['PRODUCTS'].keys():
                result['PRODUCTS'][prediction].append(text)
            else:
                result['PRODUCTS'][prediction] = [text]
        else:
            result[prediction] = text

    # result['PRODUCTS'] = process_product(products)

    return json.dumps(result, indent=4)


def reformat(text: str):
    try:
        text = text.replace('.', '').replace(',', '').replace(':', '').replace('/', '').replace('|', '').replace(
            '\\', '').replace(')', '').replace('(', '').replace('-', '').replace(';', '').replace('_', '')
        return int(text)
    except:
        return text


def process_product(products):
    label_counts = Counter([item.keys()[0] for item in products])
    max_label_counts = max(label_counts.values())
    for k, v in label_counts.items:
        if v < max_label_counts:
            products.append({k: ('', (0, 0))})

    num_of_keys = len(list(label_counts.keys()))
    chosen_key = list(label_counts.keys())[0]
    final_products = []
    i = 0
    while len(products) != 0:
        anchor = products[0].keys()[0]
        tmp = {anchor: products[0].values()[0]}
        indices_to_remove = []
        for j in range(1, len(products)):
            prediction = products[j].keys()[0]
            if prediction != anchor:
                if prediction in tmp.keys():
                    d0 = distance(tmp[anchor][1], tmp[prediction][1])
                    d1 = distance(tmp[anchor][1], products[j].values()[0][1])
                    if d0 > d1:
                        tmp[prediction] = products[j].values()[0]
                        indices_to_remove.append(j)
                else:
                    tmp[prediction] = products[j].values()[0]
                    indices_to_remove.append(j)

        final_products.append(tmp)

        for index in sorted(indices_to_remove, reverse=True):
            products.pop(index)

    return final_products


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


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
