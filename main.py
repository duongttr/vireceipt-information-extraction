import matplotlib.pyplot as plt
from Canny import *
from PIL import Image

# Load image
path = 'Data/3.jpg'
img = Image.open(path)
np_img = np.array(img).astype('uint8')

# Scanner object
scanner = Canny()

# Run
preprocess = scanner.preprocessing(np_img)
final, result = scanner.scan(preprocess)

# Image during process
scanner.plot_image(result)

# Show
plt.imshow(final)
plt.show()