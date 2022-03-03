import os
import face_recognition as face
import numpy as np


# 写在前面：这个py文件是运行不出你想要的结果的
# 不过它之中的CCA算法是没有问题的，问题在于我只是对测试的人的正脸和侧脸进行了融合，根据我询问的老师的说法：
# 进行过CCA之后，向量不再是你认为的那个可以生搬硬套的向量了，你可以稍微运行一下该文件看一下结果，
# 以便对情况有个大概的了解。真正能够运行是multiview fr2文件，那个文件中不光是对测试的人的正脸和侧脸进行了融合，
# 同时对数据库中的人的正脸和侧脸进行了融合。



#加载所有数据库图片
images = os.listdir('small_images')
#加载所有待测试的正脸图片
test_images1 = os.listdir('small_test_face1')
#加载所有待测试的侧脸图片
test_images2 = os.listdir('small_test_face2')

#笨办法，将所有数据库中的人脸手动编码
first_image=face.load_image_file('small_images/1关之琳.jpg')
first_image_encoded=face.face_encodings(first_image)[0]

second_image=face.load_image_file('small_images/2印度大叔.jpg')
second_image_encoded=face.face_encodings(second_image)[0]

third_image=face.load_image_file('small_images/3古巨基.jpg')
third_image_encoded=face.face_encodings(third_image)[0]

forth_image=face.load_image_file('small_images/4吴京.jpg')
forth_image_encoded=face.face_encodings(forth_image)[0]

fifth_image=face.load_image_file('small_images/5姚明.jpg')
fifth_image_encoded=face.face_encodings(fifth_image)[0]

sixth_image=face.load_image_file('small_images/6李冰冰.jpg')
sixth_image_encoded=face.face_encodings(sixth_image)[0]

seventh_image=face.load_image_file('small_images/7杨幂.jpg')
seventh_image_encoded=face.face_encodings(seventh_image)[0]

eightth_image=face.load_image_file('small_images/8栾风光.jpg')
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

#将数据库图片的编码一次性放入数组中
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
    fifthingth_image_encoded
]





#将所有正脸图片编码
first_image1=face.load_image_file('small_test_face1/1关之琳.jpg')
first_image_encoded1=face.face_encodings(first_image1)[0]

second_image1=face.load_image_file('small_test_face1/2印度大叔.jpg')
second_image_encoded1=face.face_encodings(second_image1)[0]

third_image1=face.load_image_file('small_test_face1/3古巨基.jpg')
third_image_encoded1=face.face_encodings(third_image1)[0]

forth_image1=face.load_image_file('small_test_face1/4吴京.jpg')
forth_image_encoded1=face.face_encodings(forth_image1)[0]

fifth_image1=face.load_image_file('small_test_face1/5姚明.jpg')
fifth_image_encoded1=face.face_encodings(fifth_image1)[0]

sixth_image1=face.load_image_file('small_test_face1/6李冰冰.jpg')
sixth_image_encoded1=face.face_encodings(sixth_image1)[0]

seventh_image1=face.load_image_file('small_test_face1/7杨幂.jpg')
seventh_image_encoded1=face.face_encodings(seventh_image1)[0]

eightth_image1=face.load_image_file('small_test_face1/8栾风光.jpg')
eightth_image_encoded1=face.face_encodings(eightth_image1)[0]

ninth_image1=face.load_image_file('small_test_face1/9甄子丹.jpg')
ninth_image_encoded1=face.face_encodings(ninth_image1)[0]

tenth_image1=face.load_image_file('small_test_face1/10胡歌.jpg')
tenth_image_encoded1=face.face_encodings(tenth_image1)[0]

eleventh_image1=face.load_image_file('small_test_face1/11范冰冰.jpg')
eleventh_image_encoded1=face.face_encodings(eleventh_image1)[0]

twelveth_image1=face.load_image_file('small_test_face1/12邓超.jpg')
twelveth_image_encoded1=face.face_encodings(twelveth_image1)[0]

thridtingth_image1=face.load_image_file('small_test_face1/13郭德纲.jpg')
thridtingth_image_encoded1=face.face_encodings(thridtingth_image1)[0]

forthingth_image1=face.load_image_file('small_test_face1/14马伊琍.jpg')
forthingth_image_encoded1=face.face_encodings(forthingth_image1)[0]

fifthingth_image1=face.load_image_file('small_test_face1/15黄晓明.jpg')
fifthingth_image_encoded1=face.face_encodings(fifthingth_image1)[0]

#将正脸图片的编码一次性放入数组中
known_faces1=[
    first_image_encoded1,
    second_image_encoded1,
    third_image_encoded1,
    forth_image_encoded1,
    fifth_image_encoded1,
    sixth_image_encoded1,
    seventh_image_encoded1,
    eightth_image_encoded1,
    ninth_image_encoded1,
    tenth_image_encoded1,
    eleventh_image_encoded1,
    twelveth_image_encoded1,
    thridtingth_image_encoded1,
    forthingth_image_encoded1,
    fifthingth_image_encoded1
]





#将所有侧脸图片编码
first_image2=face.load_image_file('small_test_face2/1关之琳.jpg')
first_image_encoded2=face.face_encodings(first_image2)[0]

second_image2=face.load_image_file('small_test_face2/2印度大叔.jpg')
second_image_encoded2=face.face_encodings(second_image2)[0]

third_image2=face.load_image_file('small_test_face2/3古巨基.jpg')
third_image_encoded2=face.face_encodings(third_image2)[0]

forth_image2=face.load_image_file('small_test_face2/4吴京.jpg')
forth_image_encoded2=face.face_encodings(forth_image2)[0]

fifth_image2=face.load_image_file('small_test_face2/5姚明.jpg')
fifth_image_encoded2=face.face_encodings(fifth_image2)[0]

sixth_image2=face.load_image_file('small_test_face2/6李冰冰.jpg')
sixth_image_encoded2=face.face_encodings(sixth_image2)[0]

seventh_image2=face.load_image_file('small_test_face2/7杨幂.jpg')
seventh_image_encoded2=face.face_encodings(seventh_image2)[0]

eightth_image2=face.load_image_file('small_test_face2/8栾风光.jpg')
eightth_image_encoded2=face.face_encodings(eightth_image2)[0]

ninth_image2=face.load_image_file('small_test_face2/9甄子丹.jpg')
ninth_image_encoded2=face.face_encodings(ninth_image2)[0]

tenth_image2=face.load_image_file('small_test_face2/10胡歌.jpg')
tenth_image_encoded2=face.face_encodings(tenth_image2)[0]

eleventh_image2=face.load_image_file('small_test_face2/11范冰冰.jpg')
eleventh_image_encoded2=face.face_encodings(eleventh_image2)[0]

twelveth_image2=face.load_image_file('small_test_face2/12邓超.jpg')
twelveth_image_encoded2=face.face_encodings(twelveth_image2)[0]

thridtingth_image2=face.load_image_file('small_test_face2/13郭德纲.jpg')
thridtingth_image_encoded2=face.face_encodings(thridtingth_image2)[0]

forthingth_image2=face.load_image_file('small_test_face2/14马伊琍.jpg')
forthingth_image_encoded2=face.face_encodings(forthingth_image2)[0]

fifthingth_image2=face.load_image_file('small_test_face2/15黄晓明.jpg')
fifthingth_image_encoded2=face.face_encodings(fifthingth_image2)[0]

#将侧脸图片的编码一次性放入数组中
known_faces2=[
    first_image_encoded2,
    second_image_encoded2,
    third_image_encoded2,
    forth_image_encoded2,
    fifth_image_encoded2,
    sixth_image_encoded2,
    seventh_image_encoded2,
    eightth_image_encoded2,
    ninth_image_encoded2,
    tenth_image_encoded2,
    eleventh_image_encoded2,
    twelveth_image_encoded2,
    thridtingth_image_encoded2,
    forthingth_image_encoded2,
    fifthingth_image_encoded2
]

X=np.array(known_faces1)
Y=np.array(known_faces2)

#---------------------------------------------------------------------------------------
#cca算法
#计算各种协方差矩阵

Z=np.concatenate((X,Y),axis=1)#将X,Y拼接
print("打印一下Z")
print(type(Z))
print(Z.shape)
C=np.cov(Z,rowvar=False) #求Z的协方差矩阵
print("打印一下C")
print(type(C))
print(C.shape)
cxx=C[0:128, 0:128]
cxy=C[0:128, 128:256]
cyx=C[128:256, 0:128]
cyy=C[128:256, 128:256]
print("打印一下协方差cxx")
print(type(cxx))
print(cxx.shape)
print("打印一下cxy")
print(type(cxy))
print(cxy.shape)
#计算G1,G2
cyy1 = np.matrix(cyy)#将数组变成矩阵类型
cxx1 = np.matrix(cxx)
cxy1 = np.matrix(cxy)
cyx1 = np.matrix(cyx)
G1 = cyy1.I * cyx1 * cxx1.I * cxy1
G2 = cxx1.I * cxy1 * cyy1.I * cyx1
print("打印一下G1的类型")
print(type(G1))
print(np.shape(G1))

#求解G1,G2 的非零特征值和特征向量

value1, xiangliang1 = np.linalg.eig(G1)
value2, xiangliang2 = np.linalg.eig(G2)

#以下是最为复杂的地方

#先处理G1的特征值和特征向量,具体步骤如下:
#1：剔除掉为0的特征值
#2：对特征值排序，降序排列
#3：注意在1,2步骤的时候，要确保特征向量的位置是同步变换的


#遍历value1数组，选出特征值不为零的特征值，记下其序号，保存到一个数组中
#先初始化一个列表，用来存放不是零的特征值的序号
index=[]
value1_after=[]#初始化特征值的list类型，用来存放非零特征值
for i, value in enumerate(value1):
    if(value)!=0:
        index.append(i)#记录当前索引值
        value1_after.append(value)#将该特征值保存到列表中
#将列表index转化为数组
index_array=np.array(index)
#将列表value1_after转化成数组，以便后面进行降序排列
value1_array=np.array(value1_after)#现在该数组中盛放的是非零的特征值


#先将向量的类型由矩阵转换成数组（一开始好像是list类型的）
xiangliang1_array=np.array(xiangliang1)
#定义一个list用来存放与索引对应的向量
xiangliang1_after=[]
for index in index_array:
    for j,xiangliang in enumerate(xiangliang1_array):
        if(j==index):
            xiangliang1_after.append(xiangliang)#将该索引对应的向量依次保存
xiangliang1_array1=np.array(xiangliang1_after)#再把list变成数组

#此时特征值和特征向量的位置仍然是一一对应的
#接下来对特征值进行降序排列，输出从大到小的特征值的索引
#再次定义一个list用来存放索引值
index_sort=np.argsort(-value1_array)#输出降序后的value_array中特征值的序号
#接下来更新向量的顺序，更新的依据就是刚刚得到的index_sort
#定义Wx_list,用来存放更新后的向量
Wx_list=[]
for index in index_sort:
    for j,xiangliang in enumerate(xiangliang1_array1):
        if j==index:
               Wx_list.append(xiangliang)
#将Wx.list转换为数组
Wx=np.array(Wx_list)

#-------------------------------------------------------------------

#再处理G2的特征值和特征向量,具体步骤如下:
#1：剔除掉为0的特征值
#2：对特征值排序，降序排列
#3：注意在1,2步骤的时候，要确保特征向量的位置是同步变换的


#遍历value2数组，选出特征值不为零的特征值，记下其序号，保存到一个数组中
#先初始化一个列表，用来存放不是零的特征值的序号
index2=[]
value2_after=[]#初始化特征值的list类型，用来存放非零特征值
for i, value in enumerate(value2):
    if(value)!=0:
        index2.append(i)#记录当前索引值
        value2_after.append(value)#将该特征值保存到列表中
#将列表index转化为数组
index2_array=np.array(index2)
#将列表value2_after转化成数组，以便后面进行降序排列
value2_array=np.array(value2_after)#现在该数组中盛放的是非零的特征值


#先将向量的类型由矩阵转换成数组（一开始好像是list类型的）
xiangliang2_array=np.array(xiangliang2)
#定义一个list用来存放与索引对应的向量
xiangliang2_after=[]
for index in index2_array:
    for j,xiangliang in enumerate(xiangliang2_array):
        if(j==index):
            xiangliang2_after.append(xiangliang)#将该索引对应的向量依次保存
xiangliang2_array2=np.array(xiangliang2_after)#再把list变成数组

#此时特征值和特征向量的位置仍然是一一对应的
#接下来对特征值进行降序排列，输出从大到小的特征值的索引
#再次定义一个list用来存放索引值
index2_sort=np.argsort(-value2_array)#输出降序后的value2_array中特征值的序号
#接下来更新向量的顺序，更新的依据就是刚刚得到的index_sort
#定义Wx_list,用来存放更新后的向量
Wy_list=[]
for index in index2_sort:
    for j,xiangliang in enumerate(xiangliang2_array2):
        if j==index:
                Wy_list.append(xiangliang)
#将Wx.list转换为数组
Wy=np.array(Wy_list)

# CCA算法结束，
# 接下来的操作和fr1.py差不多，
# 给出一个人的正脸x，让Wx*x，给出同样一个人的侧脸y，Wy*y，之后相加就是新的特征向量

#定义sum，代表识别失败的数量
sum = 0
for test_image1 in test_images1:
    print("当前识别人物：" + test_image1)
    # 预加载图片
    current_test_image1 = face.load_image_file("small_test_face1/" + test_image1)
    # 将加载的图片编码
    current_test_image1_encoded = face.face_encodings(current_test_image1)[0]
    for test_image2 in test_images2:
        if test_image2 == test_image1:
            print("已找到" + test_image1 +"的侧脸，接下来进行特征融合")
            # 预加载图片
            current_test_image2 = face.load_image_file("small_test_face2/" + test_image2)
            # 将加载的图片编码
            current_test_image2_encoded = face.face_encodings(current_test_image2)[0]
            #Wx*x,Wy*y
            current_test_image1_encoded=np.array(current_test_image1_encoded)#先变成数组
            current_test_image2_encoded=np.array(current_test_image2_encoded)
            Wz1 = np.dot(Wx,current_test_image1_encoded)
            Wz2 = np.dot(Wy,current_test_image2_encoded)
            #新向量采用相加的方法得到
            new_feature = Wz1+Wz2
            #接下来去数据库中匹配
            # 定义计数器
            count = 0
            # 先判断与数据库中几张图片相似
            distances = face.face_distance(known_faces, new_feature)
            print("已执行")
            print(type(distances))
            print(distances)
            for i in distances:
                if i < 0.5:
                    count = count + 1
            if count > 1:
                sum = sum + 1
                print("识别失败：数据库中有多人与其相似")
            elif count<1:
                print("识别失败：数据库中找不到该人物")
                sum = sum + 1
            else:
                # 循环遍历数据库中的照片
                for image in images:
                    # 加载图片
                    current_image = face.load_image_file("small_images/" + image)

                    # 将加载的图像编码为特征向量
                    current_image_encoded = face.face_encodings(current_image)[0]

                    distance = face.face_distance([new_feature], current_image_encoded)

                    if distance < 0.5:
                        if image == test_image1:
                            print("识别成功！")
                        else:
                            print("识别失败，人物不匹配")
                            sum = sum + 1

print("计算识别率")
correct_rate = (15-sum)/15*100
print("识别正确率：" + str(correct_rate) + "%")


print("end")