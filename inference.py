from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification
from ultralytics import YOLO
import pytesseract
from PIL import Image
import torch
import os
from utils import draw_output, normalize_box

# os.chmod("models\\tessdata\\vie.traineddata", 0o700)
os.environ["TESSDATA_PREFIX"] = r'models\tessdata'
pytesseract.pytesseract.tesseract_cmd = r"models\tessdata"


class LayoutLMv3:
    def __init__(self,
                 processor_pretrained=r'microsoft/layoutlmv3-base',
                 layoutlm_pretrained=r'models\checkpoint-12-mar-2023',
                 yolo_pretrained=r'models\best.pt'):
        self.processor = AutoProcessor.from_pretrained(
            processor_pretrained, apply_ocr=False)
        self.lalm_model = AutoModelForTokenClassification.from_pretrained(
            layoutlm_pretrained)
        self.yolo_model = YOLO(yolo_pretrained)
        # self.tesseract_cfg = tesseract_custom_config

    def predict(self, input_image, output_path=None):
        bboxes = self.yolo_model.predict(source=input_image, conf=0.1)[
            0].boxes.xyxy.int()
        texts = []
        normalized_boxes = []
        for box in bboxes:
            tlx, tly, brx, bry = int(box[0]), int(
                box[1]), int(box[2]), int(box[3])
            normalized_boxes.append(normalize_box(
                box, input_image.width, input_image.height))
            image_cropped = input_image.crop((tlx-3, tly-3, brx+3, bry+3))
            data = pytesseract.image_to_string(
                image_cropped, config='--oem 2 --psm 3', lang='vie', output_type=pytesseract.Output.DICT)
            texts.append(data['text'].strip())
            # image_cropped.save(f"output/test{str(box)}.jpg")

        encoding = self.processor(input_image, texts,
                                  boxes=normalized_boxes,
                                  return_offsets_mapping=True,
                                  return_tensors='pt')
        offset_mapping = encoding.pop('offset_mapping')

        with torch.no_grad():
            outputs = self.lalm_model(**encoding)

        if isinstance(output_path, str):
            os.makedirs(output_path, exist_ok=True)
            img_output = draw_output(
                image=input_image,
                logits=outputs.logits,
                id2label=self.lalm_model.config.id2label,
                label2id=self.lalm_model.config.label2id,
                token_boxes=encoding.bbox.squeeze().tolist(),
                offset_mapping=offset_mapping.squeeze().tolist()
            )

            img_output.save(os.path.join(output_path, 'result.jpg'))


if __name__ == '__main__':
    model = LayoutLMv3()
    results = model.predict('dataset/images/mcocr_public_145014zxrle_jpg.rf.667cbd346af811b44d93d38b1f14635e.jpg',
                            output_path='output')
