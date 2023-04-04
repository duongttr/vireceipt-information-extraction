import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image, ImageOps, ImageEnhance


class Transforms:

    def __init__(self, model):
        """
        The function __init__() is a special function in Python classes. It is run as soon as an object
        of a class is instantiated. The method __init__() is similar to constructors in C++ and Java

        Args:
          model: The model to be used for prediction.
        """
        self.model = model

    def _reduce_size(self, or_image, size=-1, padding=0):
        """
        It takes an image, resizes it to a given size, and then pads it with zeros to make it square

        Args:
          or_image: The original image
          size: The size of the image. If -1, the image will be resized to the max size of the image.
          padding: The amount of padding to add to the image. Defaults to 0

        Returns:
          a tuple of two images. The first image is the resized image with the padding. The second image
        is the original image with the padding.
        """
        if size == -1:
            new_img = ImageOps.expand(or_image, padding)
            return np.array(new_img).astype('uint8')

        size = size - padding
        s = or_image.size
        if s[0] <= s[1]:
            image = or_image.copy().resize((int(s[0] * (size / s[1])), size))
        else:
            image = or_image.copy().resize((size, int(s[1] * (size / s[0]))))

        new_size = image.size
        delta_w = size - new_size[0] + padding
        delta_h = size - new_size[1] + padding
        new_padding = (delta_w // 2, delta_h // 2, delta_w -
                       (delta_w // 2), delta_h - (delta_h // 2))
        new_img = ImageOps.expand(image, new_padding)

        or_size = or_image.size
        delta_w = max(or_size) - or_size[0] + padding * \
            int(max(or_image.size) / (size + padding))
        delta_h = max(or_size) - or_size[1] + padding * \
            int(max(or_image.size) / (size + padding))
        or_padding = (delta_w // 2, delta_h // 2, delta_w -
                      (delta_w // 2), delta_h - (delta_h // 2))
        or_image = ImageOps.expand(or_image, or_padding)

        return Image.fromarray(np.uint8(new_img)/255), Image.fromarray(np.uint8(or_image))

    def _order_points(self, pts):
        """
        The function takes in a list of points and returns a list of points in the following order:
        top-left, top-right, bottom-right, bottom-left

        Args:
          pts: The points that we want to order.

        Returns:
          The four points of the rectangle.
        """
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect.astype('int').tolist()

    def _find_dest(self, pts):
        """
        Given a list of points, find the point that is closest to the origin

        Args:
          pts: a list of points, each point is a list of two numbers, the first number is the x
        coordinate, the second number is the y coordinate.
        """
        (tl, tr, br, bl) = pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        destination_corners = [[0, 0], [maxWidth, 0],
                               [maxWidth, maxHeight], [0, maxHeight]]

        return self._order_points(destination_corners)

    def adaptive_binary_image(self, image, mode='mean', block_size=11, constant=2):
        """
        It takes an image, and returns a binary image where the threshold is determined by the mean or
        gaussian of the surrounding pixels

        Args:
          image: The image to be thresholded.
          mode: 'mean' or 'gaussian'. Defaults to mean
          block_size: Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
          constant: Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.

        Returns:
          The binary image is being returned.
        """
        if mode == 'mean':
            binary_image = cv2.adaptiveThreshold(np.array(image), 255,
                                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY,
                                                 block_size, constant)
        elif mode == 'gaussian':
            binary_image = cv2.adaptiveThreshold(np.array(image), 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY,
                                                 block_size, constant)

        return Image.fromarray(binary_image)

    # Dựng ảnh bị nghiêng lên
    def warp_perspective(self, or_image):
        # Reshape ảnh
        image, or_img = self._reduce_size(or_image, 256, 10)
        image, or_img = np.asarray(image), np.asarray(or_img)

        # Tạo mask
        mask = self.model.predict(image.reshape(
            (1, image.shape[0], image.shape[0], 1))).reshape((256, 256))

        # Bộ lọc cạnh
        canny = cv2.Canny(np.uint8(mask*255), 0, 200)
        # Làm các cạnh liền mạch không đứt gãy
        canny = cv2.dilate(canny, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)))
        # Xác định các cạnh
        contours, hierarchy = cv2.findContours(
            canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Chọn 5 vùng có diện tích lớn nhất
        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # Xấp xỉ các các tọa độ cạnh để tìm 4 điểm góc
        for c in page:
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            if len(corners) == 4:
                break

        # Chuyển các tọa độ góc về kích thước ban đầu để trích xuất bill từ ảnh gốc chứ không phải ảnh reshape
        corners = sorted(np.concatenate(corners).tolist())
        corners = (np.array(corners) *
                   (or_img.shape[0] / image.shape[0])).tolist()

        # Tọa độ điểm của ảnh đầu vào
        corners = self._order_points(corners)
        # Tạo độ điểm của hình muốn tham chiếu
        destination_corners = self._find_dest(corners)
        # Tiến hành warp
        M = cv2.getPerspectiveTransform(np.float32(
            corners), np.float32(destination_corners))
        warp_img = cv2.warpPerspective(or_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                       flags=cv2.INTER_LINEAR)
        return Image.fromarray(warp_img)

    def extract(self, image):
        return self.adaptive_binary_image(self.warp_perspective(image))

    def blur(self, image, blur=0):
        """
        It takes an image and a blur value, and returns a blurred image

        Args:
          image: The image to be blurred.
          blur: The amount of blur to apply to the image. 
          blur parameter is standard deviation having range in [0, Infinity], 
          Increase blur by increasing kernel size or blur parameter
        """
        blur_img = cv2.GaussianBlur(np.asarray(image), (5, 5), blur)
        return Image.fromarray(blur_img)

    def enhance_contrast(self, image, factor=1.5):
        """
        It takes an image and a factor, and returns a new image with the contrast enhanced by the given
        factor

        Args:
          image: The image to be enhanced.
          factor: A floating point value controlling the enhancement. 
          Factor 1.0 always returns a copy of the original image, 
          lower factors mean less contrast, and higher values more. 
          There are no restrictions on this value.
        """
        enhancer = ImageEnhance.Contrast(image)
        enhanced_img = enhancer.enhance(factor)
        return enhanced_img

    def enhance_sharp(self, image, factor=1.5):
        """
        It takes an image and a factor, and returns a sharpened version of the image

        Args:
          image: The image to be sharpened.
          factor: A floating point value controlling the enhancement. 
          Factor 1.0 always returns a copy of the original image, 
          lower factors mean less sharp, and higher values more. 
          There are no restrictions on this value.
        """
        enhancer = ImageEnhance.Sharpness(image)
        enhanced_img = enhancer.enhance(factor)
        return enhanced_img

    def get_canny(self, image):
        # edge extraction
        or_size = np.array(image).shape
        image, _ = self._reduce_size(image, 256, 10)
        image = np.asarray(image)
        mask = self.model.predict(image.reshape(
            (1, image.shape[0], image.shape[0], 1))).reshape((256, 256))
        canny = cv2.Canny(np.uint8(mask * 255), 0, 200)

        # Convert to origin size
        zoom = ndimage.zoom(canny, or_size[0]/image.shape[0])
        x = (zoom.shape[0] - or_size[0])//2
        y = (zoom.shape[1] - or_size[1])//2
        crop = zoom[x:(zoom.shape[0]-x), y:(zoom.shape[1]-y)]

        return Image.fromarray(crop)


class Utils():
    def __init__(self):
        pass

    def plot_results(self, results: dict, fig_size=(8, 8), rows=1):
        columns = int(len(results.keys())/rows)

        names = list(results.keys())
        fig = plt.figure(figsize=fig_size)
        for i in range(len(names)):
            subplot = fig.add_subplot(rows, columns, i+1)
            subplot.title.set_text(names[i])
            plt.imshow(results[names[i]], cmap='gray')
        plt.show()
