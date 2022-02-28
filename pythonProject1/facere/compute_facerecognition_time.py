import os
import face_recognition as face
import numpy as np
import time

start=time.time()                                                    # 记录开始的时间，以便计算花费的时间
account=0                                                            # 定义一个计数器，表示当前是几张照片了
images = os.listdir('frontal_face30')                                # 随便加载一个文件夹
known_images_encodeds_list = []                                      # 生成一个列表
known_image=face.load_image_file('frontal_face30/221.png')           # 随便加载一张照片，用来表示某个收货人的证件照
known_image_encode=face.face_encodings(known_image)[0]               # 对照片编码
known_images_encodeds_list.append(known_image_encode)                # 将编码加入到列表中
known_images_encodeds_array = np.array(known_images_encodeds_list)   # 将列表形式转化为数组形式

profile_faces = os.listdir('frontal_face30')                           # 加载所有图片，懒的改名字了，就叫profile吧
for profile_face in profile_faces:
    print("当前无人机拍摄到是" + profile_face)                              # 注意实际上是不知道无人机拍摄到的行人是谁，这里是上帝视角
    account = account + 1
    pofile_face = face.load_image_file("500face_time/" + profile_face)  # 加载该侧脸图片
    profile_faces_encodes = face.face_encodings(pofile_face)            # 对侧脸编码
    if len(profile_faces_encodes) > 0:
        profile_faces_encode = profile_faces_encodes[0]
    distance = face.face_distance(known_images_encodeds_array, profile_faces_encode)  # 计算二者之间的欧氏距离
    print('当前是第'+str(account)+'次匹配,距离为:'+str(distance))
end=time.time()
print('Total time is:'+str(end-start)+'seconds')