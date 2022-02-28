import os
import face_recognition as face
import numpy as np
#用来测试无人机随机拍摄的侧脸（也可能会包含正脸，毕竟是随机拍的）与证件照之间匹配的精度
#更新：Id_face120此时是164张重新截取的图片，其余文件夹也是如此

disaccount=0                                                         # 定义一个用来记录识别错误数量的变量
account=0
sum=0                                                                # 定义存放总量的变量


frontal_faces = os.listdir('Id_face120')                             # 加载所有收货人证件照片
frontal_faces_encode_list = []                                       # 定义列表，用来存放正脸的编码
frontal_person_name_list=[]                                          # 用来存放人的身份
for image in frontal_faces:                                          # 循环从列表中读取人脸照片
    sum=sum+1                                                        # 总量加一
    frontal_person_name_list.append(image)                           # 将人物身份同步保存
    current_face = face.load_image_file("Id_face120/" + image)       # 加载要编码的图片
    current_face_encoded = face.face_encodings(current_face)         # 编码
    frontal_faces_encode_list.append(current_face_encoded)           # 挨个儿放入list中
frontal_faces_encode_array=np.array(frontal_faces_encode_list)       # 将list类型变为array类型
print(frontal_faces_encode_array.shape)
frontal_faces_encode_array=np.squeeze(frontal_faces_encode_array)    # 将无用的1个维度压缩掉
print(frontal_faces_encode_array.shape)
print("收货人证件照片加载完成")


profile_faces = os.listdir('profile_face120')                        # 加载所有侧脸图片
for i1, frontal_person_name in enumerate(frontal_person_name_list):  # 依次对每张正脸照片寻找其侧脸
    flag=0
    print("当前收货人是：" + frontal_person_name)
    for i2, frontal_feature in enumerate(frontal_faces_encode_list):  # 按照序号去人脸编码列表里找到相应身份的人脸编码
        if i2==i1:                                                    # 如果在正脸编码列表中找到了
            print("已找到"+frontal_person_name+"的编码,接下来无人机开始从行人中寻找")
            profile_faces = np.random.permutation(profile_faces)      # 每次都随机打乱侧脸图片列表，模拟无人机随机拍摄
            for profile_face in profile_faces:
                print("当前无人机拍摄到是"+profile_face)                 #注意实际上是不知道无人机拍摄到的行人是谁，这里是上帝视角
                pofile_face = face.load_image_file("profile_face120/" + profile_face)  # 加载该侧脸图片
                profile_faces_encodes = face.face_encodings(pofile_face)               # 对侧脸编码
                if len(profile_faces_encodes) > 0:
                    profile_faces_encode = profile_faces_encodes[0]
                frontal_feature_array = np.array(frontal_feature)                      # 将正脸编码数组化
                distance = face.face_distance(frontal_feature_array, profile_faces_encode)  # 计算二者之间的欧氏距离
                if distance<0.381 or distance==0.381:
                    print(profile_face+"与"+frontal_person_name+"距离小于阈值，认为是同一个人"+"("+str(distance)+")")
                    if profile_face != frontal_person_name:
                        disaccount = disaccount + 1
                    elif profile_face == frontal_person_name:
                        account = account+1
                    break  # 提前结束收货人的搜寻
print("共有"+str(disaccount)+"个收货人识别错误")
accuracy=(sum-disaccount)/sum*100
print("侧脸-证件照识别率："+ str(round(accuracy,4)) + "%")
