import os
import face_recognition as face
import numpy as np

all_images_number = 0

# 加载所有数据库图片
images1 = os.listdir('images')

# 加载所有待测试的图片
test_images1 = os.listdir('test_face1')

#将数据库和测试的所有正脸图片编码存入列表中
data_image_encode_list = []
data_person_name_list=[]
for image1 in images1:
    all_images_number=all_images_number+1
    data_person_name_list.append(image1)         #将人物名字同步保存
    current_image = face.load_image_file("images/" + image1)
    current_image_encoded = face.face_encodings(current_image)
    data_image_encode_list.append(current_image_encoded)
data_image_encode_array=np.array(data_image_encode_list)
data_image_encode_array=np.squeeze(data_image_encode_array)

print("OK")
test_image_encode_list = []
test_person_name_list=[]
for image1 in test_images1:
    test_person_name_list.append(image1)  # 将人物名字同步保存
    current_image = face.load_image_file("test_face1/" + image1)
    current_image_encoded = face.face_encodings(current_image)
    test_image_encode_list.append(current_image_encoded)
test_image_encode_array=np.array(test_image_encode_list)
test_image_encode_array=np.squeeze(test_image_encode_array)
print("OK")
#接下来进行人脸识别
sum=0  #定义一个用来记录成功数量的变量
for i1, test_person_name in enumerate(test_person_name_list):
    print("当前识别人物：" + test_person_name)
    for i2, test_feature in enumerate(test_image_encode_list):
        if i2==i1:
            distances = face.face_distance(data_image_encode_array, test_feature)

            print("接下来打印一下"+test_person_name+"与数据库中其他人的距离")
            for d1, d1_distance in enumerate(distances):
                for d2, d2_dataset_person_name in enumerate(data_person_name_list):
                    if d2==d1:
                        print("与"+d2_dataset_person_name+"的距离为:"+str(d1_distance))

            mindistance = 10.00
            for t,distance in enumerate(distances):
                if distance <= mindistance:
                    mindistance=distance
                    flag=t
    clue=0
    for j, dataset_person_name in enumerate(data_person_name_list):
        if flag==j:
            print("数据库中与其最为匹配的人是：" + dataset_person_name)
            if dataset_person_name==test_person_name:
                sum=sum+1
                clue = 1
                print("识别成功！")
                break
    if clue == 0:
        print("识别失败")
print(str(all_images_number))
correct_rate = sum/all_images_number*100
print("单视图人脸识别率："+ str(round(correct_rate,3)) + "%")