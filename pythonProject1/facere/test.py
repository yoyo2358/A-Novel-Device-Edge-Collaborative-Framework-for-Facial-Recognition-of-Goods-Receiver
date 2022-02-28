import os
import face_recognition as face
import numpy as np

all_images_number = 0

# 加载所有训练图片
train_images1 = os.listdir('images')
train_images2 = os.listdir('images2')

# 加载所有测试的图片
images1=os.listdir('images')
images2=os.listdir('images2')
test_images1 = os.listdir('test_face1')
test_images2 = os.listdir('test_face2')

#将训练的所有正脸图片编码存入列表中
train_list1 = []
for train_image1 in train_images1:
    current_image = face.load_image_file("images/" + train_image1)
    current_image_encoded = face.face_encodings(current_image)
    train_list1.append(current_image_encoded)
X = np.array(train_list1)
X=np.squeeze(X)
print("打印一下X")
print(X.shape)

#将训练的所有正脸图片编码存入列表中
train_list2 = []
for train_image2 in train_images2:
    current_image = face.load_image_file("images2/" + train_image2)
    current_image_encoded = face.face_encodings(current_image)
    train_list2.append(current_image_encoded)
Y = np.array(train_list2)
Y=np.squeeze(Y)
print("打印一下Y")
print(Y.shape)

# 使用CCA算法构造投影矩阵
# 第一步：计算各种协方差矩阵
Z=np.concatenate((X,Y),axis=1)#将X,Y拼接
print("打印一下Z")
print(type(Z))
print(Z.shape)

# 第二步：计算Z的协方差矩阵
C=np.cov(Z,rowvar=False) #rowvar=False是让python按照列来进行运算，否则默认按行运算。
print("打印一下C")
print(type(C))
print(C.shape)

cxx=C[0:128, 0:128] # cxx是C左上角的128*128的方阵
cxy=C[0:128, 128:256] # cxy是C右上角的128*128的方阵
cyx=C[128:256, 0:128] # cyx是C左下角的128*128的方阵
cyy=C[128:256, 128:256] # cyy是C右下角的128*128的方阵
print("打印一下协方差cxx")


# 第三步：计算G1,G2
cyy1 = np.matrix(cyy) # 将数组变成矩阵类型
cxx1 = np.matrix(cxx)
cxy1 = np.matrix(cxy)
cyx1 = np.matrix(cyx)
G1 = cyy1.I * cyx1 * cxx1.I * cxy1
G2 = cxx1.I * cxy1 * cyy1.I * cyx1
print("打印一下G1的类型")
print(type(G1))
print(np.shape(G1))


# 第四步：求解G1,G2 的非零特征值和特征向量
value1, xiangliang1 = np.linalg.eig(G1)
value2, xiangliang2 = np.linalg.eig(G2)


# 第五步：处理G1的特征值和特征向量，具体步骤如下:
#1：剔除掉为0的特征值
#2：对特征值排序，降序排列
#3：注意在1,2步骤的时候，要确保特征向量的位置是同步变换的



#遍历value1数组，选出特征值不为零的特征值，记下其序号，保存到一个数组中
#先初始化一个列表，用来存放不是零的特征值的序号
index=[]
value1_list=[]#初始化特征值的list类型，用来存放非零特征值
for i, value in enumerate(value1):
    if(value)!=0:
        index.append(i)#记录当前索引值
        value1_list.append(value)#将该特征值保存到列表中
#将列表index转化为数组
index_array=np.array(index)
#将列表value1_after转化成数组，以便后面进行降序排列
value1_array=np.array(value1_list)#现在该数组中盛放的是非零的特征值

#先将向量的类型由矩阵转换成数组（一开始好像是list类型的）
xiangliang1_array=np.array(xiangliang1)
#定义一个list用来存放与索引对应的向量
xiangliang1_list=[]
for index in index_array:
    for j,xiangliang in enumerate(xiangliang1_array):
        if(j==index):
            xiangliang1_list.append(xiangliang)#将该索引对应的向量依次保存
xiangliang1_array1=np.array(xiangliang1_list)#再把list变成数组

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


# 第六步：处理G2的特征值和特征向量,具体步骤和第五步类似，如下:
#1：剔除掉为0的特征值
#2：对特征值排序，降序排列
#3：注意在1,2步骤的时候，要确保特征向量的位置是同步变换的


#遍历value2数组，选出特征值不为零的特征值，记下其序号，保存到一个数组中
#先初始化一个列表，用来存放不是零的特征值的序号
index2=[]
value2_list=[]#初始化特征值的list类型，用来存放非零特征值
for i, value in enumerate(value2):
    if(value)!=0:
        index2.append(i)#记录当前索引值
        value2_list.append(value)#将该特征值保存到列表中
#将列表index转化为数组
index2_array=np.array(index2)
#将列表value2_after转化成数组，以便后面进行降序排列
value2_array=np.array(value2_list)#现在该数组中盛放的是非零的特征值


#先将向量的类型由矩阵转换成数组（一开始好像是list类型的）
xiangliang2_array=np.array(xiangliang2)
#定义一个list用来存放与索引对应的向量
xiangliang2_list=[]
for index in index2_array:
    for j,xiangliang in enumerate(xiangliang2_array):
        if(j==index):
            xiangliang2_list.append(xiangliang)#将该索引对应的向量依次保存
xiangliang2_array2=np.array(xiangliang2_list)#再把list变成数组

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


# 定义一个存储向量的list
dataset_new_features_list=[]
# 定义一个用来同步保存人物身份的list
dataset_person_name_list=[]
print("接下来进行人脸融合(数据库一方)")
for image1 in images1:
     all_images_number=all_images_number+1
     dataset_person_name_list.append(image1)                             # 将人物身份同步保存
     current_image1 = face.load_image_file("images/" + image1)
     current_image1_encoded = face.face_encodings(current_image1)[0]
     for image2 in images2:
         if image1 == image2:
             current_image2 = face.load_image_file("images2/" + image2)
             current_image2_encoded = face.face_encodings(current_image2)[0]
             Wz1 = np.dot(Wx, current_image1_encoded)
             Wz2 = np.dot(Wy, current_image2_encoded)
             # 新向量采用相加的方法得到
             new_feature = Wz1 + Wz2
             dataset_new_features_list.append(new_feature)
dataset_new_features_array = np.array(dataset_new_features_list)

print("打印一下dataset_new_features_array")
print(dataset_new_features_array.shape)

# 定义一个存储向量的list
test_new_features_list=[]
# 定义一个用来同步保存人物身份的list
test_person_name_list=[]
print("接下来进行人脸融合(待识别一方)")
for  test_image1 in test_images1:
     test_person_name_list.append(test_image1)
     current_test_image1 = face.load_image_file("test_face1/" + test_image1)
     current_test_image1_encoded = face.face_encodings(current_test_image1)[0]
     for test_image2 in test_images2:
         if test_image1 == test_image2:
             current_test_image2 = face.load_image_file("test_face2/" + test_image2)
             current_test_image2_encoded = face.face_encodings(current_test_image2)[0]
             Wz1 = np.dot(Wx, current_test_image1_encoded)
             Wz2 = np.dot(Wy, current_test_image2_encoded)
             # 新向量采用相加的方法得到
             new_feature = Wz1 + Wz2
             test_new_features_list.append(new_feature)
test_new_features_array = np.array(test_new_features_list)

print("打印一下test_new_features_array")
print(test_new_features_array.shape)


#接下来进行人脸识别
sum=0  #定义一个用来记录成功数量的变量
for i1, test_person_name in enumerate(test_person_name_list):
    print("当前识别人物：" + test_person_name)
    for i2, test_feature in enumerate(test_new_features_list):
        if i1==i2:
            distances = face.face_distance(dataset_new_features_array, test_feature)

            print("接下来打印一下"+test_person_name+"与数据库中其他人的距离")
            for d1, d1_distance in enumerate(distances):
                for d2, d2_dataset_person_name in enumerate(dataset_person_name_list):
                    if d2==d1:
                        print("与"+d2_dataset_person_name+"的距离为:"+str(d1_distance))

            mindistance = 100.0
            for t,distance in enumerate(distances):
                if distance <= mindistance:
                    mindistance=distance
                    flag=t
    clue=0
    for j, dataset_person_name in enumerate(dataset_person_name_list):
        if flag==j:
            print("数据库中与其最为匹配的人是：" + dataset_person_name)
            if dataset_person_name==test_person_name:
                sum=sum+1
                clue = 1
                print("识别成功！")
                break
    if clue == 0:
        print("识别失败")
correct_rate = sum/all_images_number*100
print("多视图人脸识别率："+ str(round(correct_rate,3)) + "%")
