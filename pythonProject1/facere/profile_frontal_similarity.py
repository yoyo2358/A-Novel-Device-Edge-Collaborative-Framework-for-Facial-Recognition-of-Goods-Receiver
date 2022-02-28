import os
import face_recognition as face
import numpy as np

#可能不好用
all_faces_number = 0

frontal_faces = os.listdir('frontal_face30')                        # 加载所有正脸图片（收货人证件照）
frontal_faces_encode_list = []                                      # 定义列表，用来存放正脸的编码
frontal_person_name_list=[]                                         # 用来存放人的身份
for image1 in frontal_faces:                                         # 循环从列表中读取人脸照片
    print(image1)
    all_faces_number=all_faces_number+1                             # 记录图片总量
    frontal_person_name_list.append(image1)                          # 将人物身份同步保存
    current_face = face.load_image_file("frontal_face30/" + image1)  # 加载要编码的图片
    current_face_encoded = face.face_encodings(current_face)        # 编码
    frontal_faces_encode_list.append(current_face_encoded)          # 挨个儿放入list中
frontal_faces_encode_array=np.array(frontal_faces_encode_list)      # 将list类型变为array类型
print(frontal_faces_encode_array.shape)
frontal_faces_encode_array=np.squeeze(frontal_faces_encode_array)   # 将无用的1个维度压缩掉
print("收货人正面照片加载完成")


profile_faces = os.listdir('profile_face30')                        # 加载所有正脸图片（收货人证件照）
profile_faces_encode_list = []                                      # 定义列表，用来存放正脸的编码
profile_person_name_list=[]                                         # 用来存放人的身份
for image1 in profile_faces:                                         # 循环从列表中读取人脸照片
    print(image1)
    profile_person_name_list.append(image1)                          # 将人物身份同步保存
    current_face = face.load_image_file("profile_face30/" + image1)  # 加载要编码的图片
    current_face_encoded = face.face_encodings(current_face)        # 编码
    profile_faces_encode_list.append(current_face_encoded)          # 挨个儿放入list中
profile_faces_encode_array=np.array(profile_faces_encode_list)      # 将list类型变为array类型
print(profile_faces_encode_array.shape)
profile_faces_encode_array=np.squeeze(profile_faces_encode_array)   # 将无用的1个维度压缩掉
print("行人侧脸照片加载完成")

generate_faces = os.listdir('generate_face30')                       # 加载所有正脸图片（转正之后的）
generate_faces_encode_list = []                                      # 定义列表，用来存放转正人脸的编码
generate_person_name_list=[]                                         # 用来存放人的身份
for image3 in generate_faces:                                         # 循环从列表中读取人脸照片
    print(image3)
    generate_person_name_list.append(image3)                          # 将人物身份同步保存
    current_face = face.load_image_file("generate_face30/" + image3)  # 加载要编码的图片
    current_face_encoded = face.face_encodings(current_face)         # 编码
    generate_faces_encode_list.append(current_face_encoded)
generate_faces_encode_array=np.array(generate_faces_encode_list)     # 将list类型变为array类型
print(generate_faces_encode_array.shape)
generate_faces_encode_array=np.squeeze(generate_faces_encode_array)  # 将无用的1个维度压缩掉
print("转正照片加载完成")

name_list=[]
frontal_and_profile_distance_list=[]
                                                                     # 接下来进行正脸与侧脸的相似度计算
for i1, profile_person_name in enumerate(profile_person_name_list):  # 从侧脸列表中挨个儿拎儿出行人身份
    print("当前行人身份：" + profile_person_name +"接下来寻找其对应的人脸编码")
    for i2, profile_feature in enumerate(profile_faces_encode_list): # 按照序号去人脸编码列表里找到相应身份的人脸编码
        if i2==i1:                                                   # 如果在侧脸编码列表中找到了
            print("已找到"+profile_person_name+"的编码,接下来去正脸身份list去寻找对应的身份")
            for i3, frontal_person_name in enumerate(frontal_person_name_list):
                if profile_person_name==frontal_person_name:
                    print("已找到对应的正脸身份，是"+frontal_person_name+"接下来寻找其对应的人脸编码")
                    for i4,frontal_feature in enumerate(frontal_faces_encode_list):
                        if i4==i3:
                            print("已找到"+frontal_person_name+"对应的人脸编码，接下来进行相似度计算")
                            frontal_feature_array=np.array(frontal_feature)
                            distance = face.face_distance(frontal_feature_array, profile_feature)
                            print(profile_person_name+"和"+frontal_person_name+"的distance为："+str(distance))
                            frontal_and_profile_distance_list.append(distance)
                            name_list.append(frontal_person_name)