import numpy as np
import cv2
import os


def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def pv2label(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 255, axis=0)] = 1
    mask_rgb[np.all(mask_convert == 0, axis=0)] = 0

    return mask_rgb


def canny_dec(img, index):
    # files = os.listdir("D:/masks_1024")
    # for file_name in files:
    #     input_file = os.path.join("D:/masks_1024", file_name)
    #     image_output_path_last = os.path.join("D:/check/check_canny", file_name)
    #     image_output_path_last2 = os.path.join("D:/check/check_rgb", file_name)
    #
    #     img = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
        if index:
            rgb_resize = cv2.resize(img, dsize=(32, 32))
            rgb_resize_2 = cv2.resize(img, dsize=(64, 64))
            rgb_resize_3 = cv2.resize(img, dsize=(128, 128))
            # rgb_resize_4 = cv2.resize(img, dsize=(16, 16))
        else:
            rgb_resize = cv2.resize(img, dsize=(48, 48))
            rgb_resize_2 = cv2.resize(img, dsize=(96, 96))
            rgb_resize_3 = cv2.resize(img, dsize=(192, 192))
            # rgb_resize_4 = cv2.resize(img, dsize=(32, 32))

        rgb = pv2rgb(rgb_resize)
        rgb_2 = pv2rgb(rgb_resize_2)
        rgb_3 = pv2rgb(rgb_resize_3)
        # rgb_4 = pv2rgb(rgb_resize_4)

        # cv2.imwrite("/mnt/check1/img.jpg", rgb.copy())

        blurred = cv2.GaussianBlur(rgb, (11, 11), 0)
        edge = cv2.Canny(blurred, 20, 200)  # 用Canny算子提取边缘
        canny = pv2label(edge)
        #cv2.imwrite("/media/dell/DATA/lhr/lhr/canny.png", canny.copy()*100)

        blurred_2 = cv2.GaussianBlur(rgb_2, (7, 7), 0)
        edge_2 = cv2.Canny(blurred_2, 20, 200)  # 用Canny算子提取边缘
        canny_2 = pv2label(edge_2)
        #cv2.imwrite("/media/dell/DATA/lhr/lhr/canny_2.png", canny_2.copy()*100)

        blurred_3 = cv2.GaussianBlur(rgb_3, (7, 7), 0)
        edge_3 = cv2.Canny(blurred_3, 20, 200)  # 用Canny算子提取边缘
        canny_3 = pv2label(edge_3)
        #cv2.imwrite("/media/dell/DATA/lhr/lhr/canny_3.png", canny_3.copy()*100)

        # blurred_4 = cv2.GaussianBlur(rgb_4, (5, 5), 0)
        # edge_4 = cv2.Canny(blurred_4, 20, 200)  # 用Canny算子提取边缘
        # canny_4 = pv2label(edge_4)

        # cv2.imwrite("/mnt/check1/mask.png", image.copy()*100)
        # return canny, canny_2, canny_3, rgb_resize, rgb_resize_2, rgb_resize_3
        return canny, canny_2, canny_3, rgb_resize, rgb_resize_2, rgb_resize_3



#
# if __name__=="__main__":
#     # canny_dec(img)
