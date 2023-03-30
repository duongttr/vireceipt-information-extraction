import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import load_model
from Invoice_extraction import InvoiceExtraction, Utils
from PIL import Image


model_path = r"data\Invoice_Segmentation_model.h5"
img_path = r"data\mcocr_public_145014zzzej.jpg"

model = load_model(model_path, compile=False)
in_extract = InvoiceExtraction(model)
img = Image.open(img_path)

results = {}
results['0. origin'] = img
# Crop the image
results['1. warp'] = in_extract.warp_perspective(img)
# Reduce noise
results['2. blur'] = in_extract.blur(results['1. warp'])
# Enhance contrast
results['3. contrast'] = in_extract.enhance_contrast(results['2. blur'])
# Enhance sharp
results['4. sharp'] = in_extract.enhance_sharp(np.array(results['3. contrast']), 3)
# Adap mean
results['5. adap mean'] = in_extract.adaptive_binary_image(results['3. contrast'], 'mean', 15, 11)
# Adap gaussian (Cái này tốt nhất nha)
results['6. adap gaussian'] = in_extract.adaptive_binary_image(results['3. contrast'], 'gaussian', 15, 11)
# Mean blur
results['7. mean_blur'] = cv2.medianBlur(results['6. adap gaussian'], 3)

util = Utils()
util.plot_results(results, rows=2)