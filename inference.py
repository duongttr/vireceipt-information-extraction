from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification
from ultralytics import YOLO
import pytesseract
import torch
import os
from utils import draw_output, normalize_box, unnormalize_box
import numpy as np

class LayoutLMv3:
    def __init__(self,
                 processor_pretrained='microsoft/layoutlmv3-base',
                 layoutlm_pretrained='models/checkpoint-12-mar-2023',
                 yolo_pretrained='models/best.pt',
                 tessdata_pretrained='models/tessdata/'):
        self.processor = AutoProcessor.from_pretrained(processor_pretrained, apply_ocr=False)
        self.lalm_model = AutoModelForTokenClassification.from_pretrained(layoutlm_pretrained)
        self.yolo_model = YOLO(yolo_pretrained)
        self.tess_path = tessdata_pretrained
        # self.tesseract_cfg = tesseract_custom_config
    
    def predict(self, input_image, output_path=None):
        bboxes = self.yolo_model.predict(source=input_image, conf=0.1)[0].boxes.xyxy.int()
        texts = []
        normalized_boxes = []
        for box in bboxes:
            tlx,tly,brx,bry = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            normalized_boxes.append(normalize_box(box, input_image.width, input_image.height))
            image_cropped = input_image.crop((tlx-3, tly-3, brx+3, bry+3))
            data = pytesseract.image_to_string(image_cropped, config=f'--oem 2 --psm 3 --tessdata-dir {self.tess_path}', lang='vie', output_type=pytesseract.Output.DICT)
            texts.append(data['text'].strip())
        
        encoding = self.processor(input_image, texts, 
                                  boxes=normalized_boxes, 
                                  return_offsets_mapping=True, 
                                  return_tensors='pt')
        offset_mapping = encoding.pop('offset_mapping')
        
        with torch.no_grad():
            outputs = self.lalm_model(**encoding)
        
        id2label = self.lalm_model.config.id2label
        logits = outputs.logits
        token_boxes = encoding.bbox.squeeze().tolist()
        offset_mapping = offset_mapping.squeeze().tolist()
        
        predictions = logits.argmax(-1).squeeze().tolist()
        is_subword = np.array(offset_mapping)[:,0] != 0
        
        true_predictions = []
        true_boxes = []
        true_texts = []
        text_idx = 0
        for idx in range(1, len(predictions)-1):
            if not is_subword[idx]:
                true_predictions.append(id2label[predictions[idx]])
                true_boxes.append(unnormalize_box(token_boxes[idx], input_image.width, input_image.height))
                true_texts.append(texts[text_idx])
                text_idx += 1
        
        if isinstance(output_path, str):
            os.makedirs(output_path, exist_ok=True)
            img_output = draw_output(
                image=input_image,
                true_predictions=true_predictions,
                true_boxes=true_boxes
            )
            img_output.save(os.path.join(output_path,'result.jpg'))
        
        final_results = []
        for box, text, label in zip(true_boxes, texts, true_predictions):
            final_results.append({
                'box': box,
                'text': text,
                'label': label
            })
        return final_results
        

if __name__ == '__main__':
    model = LayoutLMv3()
    
    from PIL import Image
    input_image = Image.open('/Users/jaydentran1909/Pet Projects/bill-information-extraction/dataset/images/mcocr_public_145014ymhqd_jpg.rf.412ad8d143240d95330b7c55d83bc95f.jpg')
    results = model.predict(input_image, output_path='output')
    print(results)