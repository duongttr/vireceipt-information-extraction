import matplotlib.pyplot as plt
from keras.models import load_model
from Invoice_extraction import InvoiceExtraction, Utils
from PIL import Image


model_path = r"data\Invoice_Segmentation_model.h5"
img_path = r"data\mcocr_public_145014wgjvg.jpg"

model = load_model(model_path, compile=False)
in_extract = InvoiceExtraction(model)
img = Image.open(img_path)

results = {}
results['0. origin'] = img
# Crop the image
results['1. warp'] = in_extract.warp_perspective(img_path)
# Reduce noise
results['2. blur'] = in_extract.blur(results['1. warp'])
# Enhance contrast
results['3. enhanced'] = in_extract.enhance(results['2. blur'])
results['4. bw'] = in_extract.adaptive_binary_image(results['3. enhanced'])


util = Utils()
util.plot_results(results)