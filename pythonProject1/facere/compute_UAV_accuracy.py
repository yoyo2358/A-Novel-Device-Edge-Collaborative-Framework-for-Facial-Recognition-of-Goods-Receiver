import os
import random

import face_recognition as face
import numpy as np

#用来测试无人机拍摄的正脸与证件照匹配的精度

disaccount=0                                                            # 定义一个用来记录识别错误数量的变量
account=0                                                            # 定义一个用来记录识别正确数量的变量
sum=0                                                                # 定义存放总量的变量


frontal_faces = os.listdir('Id_face120')                        # 加载所有正脸图片（收货人证件照）
frontal_faces_encode_list = []                                       # 定义列表，用来存放正脸的编码
frontal_person_name_list=[]                                          # 用来存放人的身份
for image in frontal_faces:                                          # 循环从列表中读取人脸照片
    sum=sum+1                                                        # 总量加一
    frontal_person_name_list.append(image)                           # 将人物身份同步保存
    current_face = face.load_image_file("Id_face120/" + image)  # 加载要编码的图片
    current_face_encoded = face.face_encodings(current_face)         # 编码
    frontal_faces_encode_list.append(current_face_encoded)           # 挨个儿放入list中
frontal_faces_encode_array=np.array(frontal_faces_encode_list)       # 将list类型变为array类型
print(frontal_faces_encode_array.shape)
frontal_faces_encode_array=np.squeeze(frontal_faces_encode_array)    # 将无用的1个维度压缩掉
print(frontal_faces_encode_array.shape)
print("收货人证件照片加载完成")


# uav_faces = os.listdir('UAV_face200')                        # 加载所有正脸图片（UAVface里面的照片相当于和证件照相似的正脸照片）
# for i1, frontal_person_name in enumerate(frontal_person_name_list):  # 依次对每张正脸照片寻找其侧脸
#     flag=0
#     print("当前收货人是：" + frontal_person_name)
#     for i2, frontal_feature in enumerate(frontal_faces_encode_list):  # 按照序号去人脸编码列表里找到相应身份的人脸编码
#         if i2==i1:                                                    # 如果在正脸编码列表中找到了
#             print("已找到"+frontal_person_name+"的编码,接下来无人机开始从行人中寻找")
#             uav_faces = np.random.permutation(uav_faces)      # 每次都随机打乱正脸图片列表，模拟无人机随机拍摄
#             for uav_face in uav_faces:
#                 print("当前无人机拍摄到是"+uav_face)                 #注意实际上是不知道无人机拍摄到的行人是谁，这里是上帝视角
#                 uav_img = face.load_image_file("UAV_face200/" + uav_face)  # 加载该正脸图片
#                 uav_img_encodes = face.face_encodings(uav_img)               # 对正脸编码
#                 if len(uav_img_encodes) > 0:
#                     uav_img_encode=uav_img_encodes[0]
#                 frontal_feature_array = np.array(frontal_feature)                      # 将正脸编码数组化
#                 distance = face.face_distance(frontal_feature_array, uav_img_encode)  # 计算二者之间的欧氏距离
#                 if distance<0.6 or distance==0.6:
#                     print(uav_face+"与"+frontal_person_name+"距离小于阈值，认为是同一个人"+"("+str(distance)+")")
#                     if uav_face==frontal_person_name:
#                         account = account + 1
#                     flag = 1
#                     break  # 提前结束收货人的搜寻
#             if flag==0:
#                 print("没有从行人中找到该收货人")
# print("共有"+str(account)+"个收货人识别正确")
# accuracy=account/sum*100
# print("无人机拍摄-正脸识别率："+ str(round(accuracy,3)) + "%")

uav_faces = os.listdir('frontal_face120')                        # 加载所有正脸图片（UAVface里面的照片相当于和证件照相似的正脸照片）
for i1, frontal_person_name in enumerate(frontal_person_name_list):  # 依次选取每张正脸，相当于挨个为收货人配送
    flag=0
    print("当前收货人是：" + frontal_person_name)
    for i2, frontal_feature in enumerate(frontal_faces_encode_list):  # 按照序号去人脸编码列表里找到相应身份的人脸编码
        if i2==i1:                                                    # 如果在正脸编码列表中找到了
            print("已找到"+frontal_person_name+"的编码,接下来无人机开始从行人中寻找")
            temporary_uav_faces = os.listdir('frontal_face120')  # 临时变量,主要用于删除指定的收货人
            pre_uav_faces=temporary_uav_faces.remove(frontal_person_name) #先把当前收货人的名字去掉
            random_list=random.sample(temporary_uav_faces,39) #随机从临时变量里面取除了收货人之外的另外39个路人的名字
            random_list.append(frontal_person_name)#再把收货人名字加进去，凑出40个人，表示姿势识别过滤之后剩下的三分之一
            uav_faces = random_list
            #uav_faces = np.random.permutation(uav_faces)      # 每次都随机打乱正脸图片列表，模拟无人机随机拍摄
            for uav_face in uav_faces:
                print("当前无人机拍摄到是"+uav_face)                 #注意实际上是不知道无人机拍摄到的行人是谁，这里是上帝视角
                uav_img = face.load_image_file("frontal_face120/" + uav_face)  # 加载该正脸图片
                uav_img_encodes = face.face_encodings(uav_img)               # 对正脸编码
                if len(uav_img_encodes) > 0:
                    uav_img_encode=uav_img_encodes[0]
                frontal_feature_array = np.array(frontal_feature)                      # 将正脸编码数组化
                distance = face.face_distance(frontal_feature_array, uav_img_encode)  # 计算二者之间的欧氏距离
                if distance<0.381 or distance==0.381:
                    print(uav_face+"与"+frontal_person_name+"距离小于阈值，认为是同一个人"+"("+str(distance)+")")
                    if uav_face!=frontal_person_name:
                        disaccount = disaccount + 1
                    elif uav_face==frontal_person_name:
                        account = account +1
                    break
print("共有"+str(disaccount)+"个收货人识别错误")
accuracy=(sum-disaccount)/sum*100
print("正脸-证件照识别率："+ str(round(accuracy,4)) + "%")