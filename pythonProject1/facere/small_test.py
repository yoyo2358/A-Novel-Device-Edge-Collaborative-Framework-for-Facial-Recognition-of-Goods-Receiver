import os
import face_recognition as face
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
start=time.time()
images = os.listdir('Id_face120')
# #定义一个列表存储数据库中的人脸编码
known_images_encodeds_list = []
# for image in images:
#     current_image = face.load_image_file("test_images1/" + image)
#     current_image_encoded = face.face_encodings(current_image)[0]
#     known_images_encodeds_list.append(current_image_encoded)
#
# img1=Image.open('frontal_face30/29.jpg')
# img2=Image.open('profile_face30/29.jpg')
# img3=Image.open('generate_face30/29.jpg')
# img1=img1.convert('LA')
# img2=img2.convert('LA')
# img3=img3.convert('LA')
# plt.imshow(img3)
# plt.show()

known_image=face.load_image_file('Id_face120/199.PNG')
known_image_encode=face.face_encodings(known_image)[0]
known_images_encodeds_list.append(known_image_encode)
# 将列表形式转化为数组形式
known_images_encodeds_array = np.array(known_images_encodeds_list)
# 加载需要要对比的图片
first_images=face.load_image_file('frontal_face120/199.PNG')
# 将加载的图像编码为特征向量
if len(first_images)>0:
    image_to_be_matched_encoded = face.face_encodings(first_images)[0]
distances=face.face_distance(known_images_encodeds_array,image_to_be_matched_encoded)
print('两者之间的相似度（距离）为'+str(distances))
end=time.time()
print('单次人脸识别的用时为：'+str(end-start)+'seconds')