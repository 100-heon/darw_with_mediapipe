import os
import cv2
import numpy as np


data_dir = 'data'

augmented_data_dir = 'data_augmented_3'


rotation_angles = [-60, -45, -30, -15, 15, 30, 45, 60]


if not os.path.exists(augmented_data_dir):
    os.makedirs(augmented_data_dir)


for label in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label)
    augmented_label_path = os.path.join(augmented_data_dir, label)


    if not os.path.exists(augmented_label_path):
        os.makedirs(augmented_label_path)


    for file_name in os.listdir(label_path):
        image_path = os.path.join(label_path, file_name)
        image = cv2.imread(image_path)

        if image is None:
            continue


        base_name, ext = os.path.splitext(file_name)
        cv2.imwrite(os.path.join(augmented_label_path, f"{base_name}_orig{ext}"), image)


        flipped_image = cv2.flip(image, 1)  
        cv2.imwrite(os.path.join(augmented_label_path, f"{base_name}_flipped{ext}"), flipped_image)


        for angle in rotation_angles:

            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
            rotated_flipped_image = cv2.warpAffine(flipped_image, rotation_matrix, (w, h))


            cv2.imwrite(os.path.join(augmented_label_path, f"{base_name}_rot{angle}{ext}"), rotated_image)

            cv2.imwrite(os.path.join(augmented_label_path, f"{base_name}_flipped_rot{angle}{ext}"), rotated_flipped_image)

print("데이터 증강 완료")
