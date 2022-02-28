import os
import face_recognition as face
import numpy as np

# 该py文件是刚开始尝试的时候所写，后期又进行了修改。
# 主要是尝试只测试一个人的一张人脸，跑一下结果看看。

# 列出数据库中所有可用的图片
images = os.listdir('small_images')
#定义一个列表存储数据库中的人脸编码
known_images_encodeds_list = []

#笨办法，将所有人脸手动编码
# first_image=face.load_image_file('small_images/1关之琳.jpg')
# first_image_encoded=face.face_encodings(first_image)[0]
#
# second_image=face.load_image_file('small_images/2印度大叔.jpg')
# second_image_encoded=face.face_encodings(second_image)[0]
#
# third_image=face.load_image_file('small_images/3古巨基.jpg')
# third_image_encoded=face.face_encodings(third_image)[0]
#
# forth_image=face.load_image_file('small_images/4吴京.jpg')
# forth_image_encoded=face.face_encodings(forth_image)[0]
#
# fifth_image=face.load_image_file('small_images/5姚明.jpg')
# fifth_image_encoded=face.face_encodings(fifth_image)[0]
#
# sixth_image=face.load_image_file('small_images/6李冰冰.jpg')
# sixth_image_encoded=face.face_encodings(sixth_image)[0]
#
# seventh_image=face.load_image_file('small_images/7杨幂.jpg')
# seventh_image_encoded=face.face_encodings(seventh_image)[0]
#
# eightth_image=face.load_image_file('small_images/8栾风光.jpg')
# eightth_image_encoded=face.face_encodings(eightth_image)[0]
#
# ninth_image=face.load_image_file('small_images/9甄子丹.jpg')
# ninth_image_encoded=face.face_encodings(ninth_image)[0]
#
# tenth_image=face.load_image_file('small_images/10胡歌.jpg')
# tenth_image_encoded=face.face_encodings(tenth_image)[0]
#
# eleventh_image=face.load_image_file('small_images/11范冰冰.jpg')
# eleventh_image_encoded=face.face_encodings(eleventh_image)[0]
#
# twelveth_image=face.load_image_file('small_images/12邓超.jpg')
# twelveth_image_encoded=face.face_encodings(twelveth_image)[0]
#
# thridtingth_image=face.load_image_file('small_images/13郭德纲.jpg')
# thridtingth_image_encoded=face.face_encodings(thridtingth_image)[0]
#
# forthingth_image=face.load_image_file('small_images/14马伊琍.jpg')
# forthingth_image_encoded=face.face_encodings(forthingth_image)[0]
#
# fifthingth_image=face.load_image_file('small_images/15黄晓明.jpg')
# fifthingth_image_encoded=face.face_encodings(fifthingth_image)[0]

#将编码一次性放入数组中
# known_images_encodeds_list=[
#     first_image_encoded,
#     second_image_encoded,
#     third_image_encoded,
#     forth_image_encoded,
#     fifth_image_encoded,
#     sixth_image_encoded,
#     seventh_image_encoded,
#     eightth_image_encoded,
#     ninth_image_encoded,
#     tenth_image_encoded,
#     eleventh_image_encoded,
#     twelveth_image_encoded,
#     thridtingth_image_encoded,
#     forthingth_image_encoded,
#     fifthingth_image_encoded
# ]
# 使用for循环将人脸编码存入list列表中
for image in images:
    current_image = face.load_image_file("small_images/" + image)
    current_image_encoded = face.face_encodings(current_image)[0]
    known_images_encodeds_list.append(current_image_encoded)
#将列表形式转化为数组形式
known_images_encodeds_array=np.array(known_images_encodeds_list)
print(known_images_encodeds_array.shape)
# 加载需要要对比的图片
first_image=face.load_image_file('small_test_face1/12邓超.jpg')
# 将加载的图像编码为特征向量
image_to_be_matched_encoded = face.face_encodings(first_image)[0]

distances=face.face_distance(known_images_encodeds_array,image_to_be_matched_encoded)
print(distances)
count=0
for distance in distances:
    if distance<0.6:
        count=count+1
if count>1:
    print("，识别失败：数据库中有多人与其相似")
elif count<0:
    print("识别失败：数据库中查无此人")
else:
    for image in images:
        current_image = face.load_image_file("small_images/" + image)
        current_image_encoded = face.face_encodings(current_image)
        distance=face.face_distance(current_image_encoded,image_to_be_matched_encoded)
        if distance<0.6:
            if image==first_image:
                print("识别成功")
            else:
                print("识别失败，人物不匹配")
                sum = sum + 1
print("end")