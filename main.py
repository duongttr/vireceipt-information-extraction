import matplotlib.pyplot as plt

from keras.models import load_model
from Invoice_extraction import InvoiceExtraction

model_path = "D:\Downloads\Invoice_Segmentation_model.h5"
img_path = "D:\Downloads\\bhx_7a28afe540a186ffdfb0.jpg"

model = load_model(model_path, compile=False)
in_extract = InvoiceExtraction(model)

result = in_extract.extract(img_path)
plt.imshow(result)
plt.show()

