import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from matplotlib import pyplot as plt

class InvoiceExtraction:

    # Import model segmentation
    def __init__(self, model):
        self.model = model

    # Chuyển hình ảnh về hình vuông kích thước 256x256 bằng các thêm padding
    def reduce_size(self, path, size=-1, padding=0):
        or_image = Image.open(path)
        or_image = ImageOps.grayscale(or_image)
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
        new_padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        new_img = ImageOps.expand(image, new_padding)

        or_size = or_image.size
        delta_w = max(or_size) - or_size[0] + padding * int(max(or_image.size) / (size + padding))
        delta_h = max(or_size) - or_size[1] + padding * int(max(or_image.size) / (size + padding))
        or_padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        or_image = ImageOps.expand(or_image, or_padding)

        return np.uint8(new_img)/255, np.uint8(or_image)

    def drawRectangle(img, biggest, thickness):
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0),
                 thickness)
        cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0),
                 thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0),
                 thickness)
        cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0),
                 thickness)

        return img

    # Sắp xếp các tọa độ của ảnh
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect.astype('int').tolist()

    # Xác định tọa độ mà ta muốn tham chiếu tới (Cụ thể là ta muốn tham chiều đầu vào về hình chữ nhật)
    def find_dest(self, pts):
        (tl, tr, br, bl) = pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

        return self.order_points(destination_corners)

    # Chuyển ảnh về trắng đen
    def thresh_hold(self, image, threshold = 175):
        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        (T, thresh) = cv2.threshold(blurred, threshold, 255,
                                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return thresh


    def adaptive_binary_image(self, image, block_size=11, constant=2):
        image = np.array(image)
        binary_image = cv2.adaptiveThreshold(image, 255, 
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 
                                             block_size, constant)
        return binary_image



    # Dựng ảnh bị nghiêng lên
    def warp_perspective(self, image_path):
        # Reshape ảnh
        image, or_img = self.reduce_size(image_path, 256, 10)

        # Tạo mask
        mask = self.model.predict(image.reshape((1, image.shape[0], image.shape[0], 1))).reshape((256, 256))

        # Bộ lọc cạnh
        canny = cv2.Canny(np.uint8(mask*255), 0, 200)
        # Làm các cạnh liền mạch không đứt gãy
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # Xác định các cạnh
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
        corners = (np.array(corners) * (or_img.shape[0] / image.shape[0])).tolist()

        # Tọa độ điểm của ảnh đầu vào
        corners = self.order_points(corners)
        # Tạo độ điểm của hình muốn tham chiếu
        destination_corners = self.find_dest(corners)
        # Tiến hành warp
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        warp_img = cv2.warpPerspective(or_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                    flags=cv2.INTER_LINEAR)
        return warp_img


    # Trích xuất Bill ra, gồm WarpPerspective và Threshold(chuyển trắng đen)
    def extract(self, image_path):
        warp_img = self.warp_perspective(image_path)
        extracted_img = self.thresh_hold(warp_img)
        return extracted_img


    # Blur, reduce noise
    def blur(self, image_path, blur=0):
        if isinstance(image_path, np.ndarray):
            image = image_path
        else:
            image, or_img = self.reduce_size(image_path, 256, 10)
        blur_img = cv2.GaussianBlur(image   , (5, 5), blur)
        return blur_img

    # Enhance the contrast, to balance with the blur
    def enhance(self, image_path, factor=1.5):
        if isinstance(image_path, np.ndarray):
            image = image_path
        else:
            image, or_img = self.reduce_size(image_path, 256, 10)
        enhancer = ImageEnhance.Contrast(Image.fromarray(image))
        enhanced_img = enhancer.enhance(factor)
        return enhanced_img

class Utils():
    def __init__(self):
        pass
    def plot_results(self, results: dict, fig_size = (8,8), rows=1):
        columns = int(len(results.keys())/rows)
        
        names = list(results.keys())
        fig = plt.figure(figsize=fig_size)
        for i in range(len(names)):        
            subplot = fig.add_subplot(rows, columns, i+1)
            subplot.title.set_text(names[i])
            plt.imshow(results[names[i]], cmap='gray')
        plt.show()