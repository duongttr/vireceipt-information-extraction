import os
import PIL.Image as PImage
from transformers import AutoProcessor
from datasets import ClassLabel, Image, load_dataset

ner_tags_list = [cls.strip() for cls in open(
    os.path.join('src', 'classes.txt'), 'r').readlines()]

id2label = {k: v for k, v in enumerate(ner_tags_list)}
label2id = {v: k for k, v in enumerate(ner_tags_list)}

class_label = ClassLabel(names=ner_tags_list)
processor = AutoProcessor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False)


class LayoutLMv3Dataset:
    def __init__(self, data_path_dict: dict):
        self.data_path_dict = data_path_dict
        self.image_column_name = "image"
        self.text_column_name = "tokens"
        self.boxes_column_name = "bboxes"
        self.label_column_name = "ner_tags"
        self.image_feat_decoder = Image().decode_example

    def get_dataset(self):
        dataset = load_dataset('json', data_files=self.data_path_dict)
        dataset = dataset.map(self.__mapping, num_proc=4,
                              remove_columns=['size'])

        column_names = dataset["train"].column_names

        train_dataset = dataset["train"].map(
            self.__prepare_examples,
            remove_columns=column_names
        )

        eval_dataset = dataset["val"].map(
            self.__prepare_examples,
            remove_columns=column_names
        )

        return {'train': train_dataset, 'val': eval_dataset}

    def __prepare_examples(self, examples):
        images = self.image_feat_decoder(examples[self.image_column_name])
        words = examples[self.text_column_name]
        boxes = examples[self.boxes_column_name]
        word_labels = examples[self.label_column_name]
        encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                             truncation=True, padding="max_length")

        encoding['pixel_values'] = encoding['pixel_values'][0]
        return encoding

    def __mapping(self, examples, root_folder='dataset/images'):
        new_bboxes = []
        # map bboxes
        for size, bboxes in zip(examples['size'], examples['bboxes']):
            W, H = size
            tlx, tly, brx, bry = bboxes
            new_bboxes.append([int(tlx / W * 1000), int(tly / H * 1000),
                              int(brx / W * 1000), int(bry / H * 1000)])
        examples['bboxes'] = new_bboxes

        # map image
        examples['image'] = PImage.open(
            os.path.join(root_folder, examples['image'])
        ).convert('RGB')

        # ner tags convert
        examples['ner_tags'] = [class_label.str2int(
            tag) for tag in examples['ner_tags']]

        return examples


if __name__ == "__main__":
    dataset = LayoutLMv3Dataset(
        data_path_dict={'train': 'dataset/train.json', 'val': 'dataset/val.json'})
    print(dataset.get_dataset())
