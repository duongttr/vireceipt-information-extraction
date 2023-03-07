import cv2
import numpy as np
from PIL import Image, ImageOps


class InvoiceExtraction:
    def __init__(self, model):
        self.model = model

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

        return np.array(new_img)/255, np.array(or_image)/255

    def reorder(self, myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)

        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]

        return myPointsNew

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

    def restore_coor(self, contours, ratio):
        (tl, tr, br, bl) = contours
        # Finding the maximum width.
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Finding the maximum height.
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        x = tl[0] * ratio // 2
        y = tl[1] * ratio // 2
        print(maxWidth, maxHeight)
        # Final destination co-ordinates.
        destination_corners = [[y, x], [x + maxWidth * ratio, y], [x + maxWidth * ratio, y + maxHeight * ratio],
                               [x, y + maxHeight * ratio]]
        return destination_corners

    def extract(self, image_path):
        image, or_img = self.reduce_size(image_path, 256, 10)
        mask = self.model.predict(image.reshape((1, image.shape[0], image.shape[0], 1))).reshape((256, 256))
        canny = cv2.Canny(np.uint8(mask*255), 0, 200)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for c in page:
            epsilon = 0.02 * cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            if len(corners) == 4:
                break

        # print(corners, int(or_img.shape[0]/image.shape[0]))
        corners = sorted(np.concatenate(corners).tolist())
        corners = (np.array(corners) * (or_img.shape[0] / image.shape[0])).tolist()
        corners = self.order_points(corners)
        destination_corners = self.find_dest(corners)
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        final = cv2.warpPerspective(or_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                    flags=cv2.INTER_LINEAR)
        
        return final
