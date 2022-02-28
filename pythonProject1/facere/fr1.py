import os
import face_recognition as face
import numpy as np

# 该py文件是fr文件的进阶版本，是测试多个人的单张人脸。
# 最终的结果是给出每个人与数据库中所有人脸的距离。并且
# 给出最终的识别结果，每个人到底是识别成功还是识别失败。

# 列出数据库中所有可用的图片
images = os.listdir('small_images')

#笨办法，将所有数据库中的人脸手动编码
first_image=face.load_image_file('small_images/1关之琳.jpg')
first_image_encoded=face.face_encodings(first_image)[0]

second_image=face.load_image_file('small_images/2佟大为.jpg')
second_image_encoded=face.face_encodings(second_image)[0]

third_image=face.load_image_file('small_images/3古巨基.jpg')
third_image_encoded=face.face_encodings(third_image)[0]

forth_image=face.load_image_file('small_images/4吴京.jpg')
forth_image_encoded=face.face_encodings(forth_image)[0]

fifth_image=face.load_image_file('small_images/5姚明.jpg')
fifth_image_encoded=face.face_encodings(fifth_image)[0]

sixth_image=face.load_image_file('small_images/6冯巩.jpg')
sixth_image_encoded=face.face_encodings(sixth_image)[0]

seventh_image=face.load_image_file('small_images/7巩汉林.jpg')
seventh_image_encoded=face.face_encodings(seventh_image)[0]

eightth_image=face.load_image_file('small_images/8李云龙.jpg')
eightth_image_encoded=face.face_encodings(eightth_image)[0]

ninth_image=face.load_image_file('small_images/9甄子丹.jpg')
ninth_image_encoded=face.face_encodings(ninth_image)[0]

tenth_image=face.load_image_file('small_images/10胡歌.jpg')
tenth_image_encoded=face.face_encodings(tenth_image)[0]

eleventh_image=face.load_image_file('small_images/11范冰冰.jpg')
eleventh_image_encoded=face.face_encodings(eleventh_image)[0]

twelveth_image=face.load_image_file('small_images/12邓超.jpg')
twelveth_image_encoded=face.face_encodings(twelveth_image)[0]

thridtingth_image=face.load_image_file('small_images/13郭德纲.jpg')
thridtingth_image_encoded=face.face_encodings(thridtingth_image)[0]

forthingth_image=face.load_image_file('small_images/14马伊琍.jpg')
forthingth_image_encoded=face.face_encodings(forthingth_image)[0]

fifthingth_image=face.load_image_file('small_images/15黄晓明.jpg')
fifthingth_image_encoded=face.face_encodings(fifthingth_image)[0]

sixteenth_image=face.load_image_file('small_images/16刘亦菲.jpg')
sixteenth_image_encoded=face.face_encodings(sixteenth_image)[0]

seventeenth_image=face.load_image_file('small_images/17周杰伦.jpg')
seventeenth_image_encoded=face.face_encodings(seventeenth_image)[0]

eightteenth_image=face.load_image_file('small_images/18杰森斯坦森.jpg')
eightteenth_image_encoded=face.face_encodings(eightteenth_image)[0]

nineteenth_image=face.load_image_file('small_images/19梁朝伟.jpg')
nineteenth_image_encoded=face.face_encodings(nineteenth_image)[0]

twenty_image=face.load_image_file('small_images/20阿汤哥.jpg')
twenty_image_encoded=face.face_encodings(twenty_image)[0]

twentyone_image=face.load_image_file('small_images/AF0301.jpg')
twentyone_image_encoded=face.face_encodings(twentyone_image)[0]

twentytwo_image=face.load_image_file('small_images/AF0302.jpg')
twentytwo_image_encoded=face.face_encodings(twentytwo_image)[0]

#将编码一次性放入数组中
known_faces=[
    first_image_encoded,
    second_image_encoded,
    third_image_encoded,
    forth_image_encoded,
    fifth_image_encoded,
    sixth_image_encoded,
    seventh_image_encoded,
    eightth_image_encoded,
    ninth_image_encoded,
    tenth_image_encoded,
    eleventh_image_encoded,
    twelveth_image_encoded,
    thridtingth_image_encoded,
    forthingth_image_encoded,
    fifthingth_image_encoded,
    sixteenth_image_encoded,
    seventeenth_image_encoded,
    eightteenth_image_encoded,
    nineteenth_image_encoded,
    twenty_image_encoded,
    twentyone_image_encoded,
    twentytwo_image_encoded
]


#加载所有待测试的图片，注意这里仅仅是加载了所有待测试人的正脸，并没有涉及到侧脸
test_images = os.listdir('small_test_face1')

#定义一个变量用来统计识别失败的个数
sum=0

for test_image in test_images:
    print("当前识别人物：" + test_image)
    # 预加载图片
    current_test_image = face.load_image_file("small_test_face1/" + test_image)
    # 将加载的图片编码
    current_test_image_encoded = face.face_encodings(current_test_image)[0]

    # 定义计数器，统计与数据库中几个人的距离小于阈值
    count=0

    # 先判断与数据库中几张图片相似
    distances = face.face_distance(known_faces, current_test_image_encoded)
    print("打印" + test_image + "与数据库中所有人之间的距离")
    print(distances)

    for i in distances:
        if i < 0.6:
            count = count + 1
    if count > 1:
        sum = sum + 1
        print("识别失败：数据库中有多人与其相似")
    elif count < 1:
        print("识别失败：数据库中找不到该人物")
        sum = sum + 1
    else:
        # 循环遍历数据库中的照片
        for image in images:
           # 加载图片
           current_image = face.load_image_file("small_images/" + image)

           # 将加载的图像编码为特征向量
           current_image_encoded = face.face_encodings(current_image)[0]

           #判断当前数据库中的图片与当前待测试图片的距离
           distance=face.face_distance([current_test_image_encoded], current_image_encoded)

           if distance<0.6:
               if image==test_image:
                   print("识别成功！")
               else:
                   print("识别失败，人物不匹配")
                   sum = sum+1

print("计算识别率")
correct_rate = (20-sum)/20*100
print("识别正确率：" + str(round(correct_rate,2)) + "%")

