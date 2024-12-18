# 这是对TensorFlow的Keras的学习过程，完全基于TensorFlow的官网内容。在学习完后，我将使用逆水寒手游的内功系统来作为自己的实验对象
# 首先，TensorFlow是一个机器学习的平台，而我们使用的keras是使用神经网络的模型。在官网上，它是用一个衣服鞋子分类来进行介绍的，我们来看看
# https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn
# TensorFlow and tf.keras 这个tf.keras是一个学习用的API
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)
# 以上是开头准备啊，我这个地方版本显示2.18.0
# 然后他用的是一个衣服鞋子包包啥的分类数据集。Fashion MNIST。每个衣服鞋子都是28*28的灰度图，10个类别，70000个图像。
# 预计使用60000个图来训练，10000个图来看看效果如何
# OK啊，导入数据库
fashion_mnist = tf.keras.datasets.fashion_mnist
# 下面这个代码你可以看到，很奇怪，咋自动分了两个集呢？实际上这个数据集在创作时就已经分成了60000和10000的训练集和测试集，这个fashion_mnist.load_data()就是加载训练集，然后分别放入两个不同的集中。
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 我们试试别的行不行，如果只放到一个集中？会怎样？我试试下面的代码。好像也能跑，那我们看看这个数据集的结构是什么样的。

# //?(images, labels) = fashion_mnist.load_data()
# //?print(images.shape)
# //?print(labels.shape)

# 很遗憾！结果是这样的：
# //!“Traceback (most recent call last):
#   //!File "c:\Users\64171\Desktop\TensorFlow入门.py", line 17, in <module>
#   //!  print(images.shape)
#      //!     ^^^^^^^^^^^^
# //!AttributeError: 'tuple' object has no attribute 'shape'”

# 哇，所以咱们就明白了，实际上，fashion_mnist.load_data返回的是一个元组，这个元组大概长这样“((train_images,train_labels), (test_images,test_labels))”。我们可以看到，这是一个元组嵌套两个元组，所以，如果
# 我们直接赋给(images, labels)，那就变成images接了一个元组(train_images,train_labels)，labels接了一个元组(test_images,test_labels)，这样一搞，那就爆炸了，所以，咱们还是得按前面的接法，(train_images, 
# train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 好吧，我们把上面注释掉，跟着入门向导继续。

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape) #(60000, 28, 28)
print(train_labels.shape) #(60000,)
print(test_images.shape) #(10000, 28, 28)
print(test_labels.shape) #(10000,)

# 很好啊很好，我们看到上面的结果了，很成功啊！这个image就是图像，第一个数字代表有多少数据，第二个和第三个数字就是这个图像的结构，28*28像素,我们来进一步探究探究这个数据集到底是个啥东西。
print(train_labels[1:10])  # 看看前 10 个标签
print(train_images[0])  # 看看第一个
print(train_images[0].shape) #看看第一个数组到底是什么样子
print(np.unique(train_labels))  # 看看有哪些值

#?结果如下啊：太长了，我还是用绿色来注释吧。
# 这是第一个，我们看看前十个标签是啥子，哦，是前十个图片所属的类别。
# [0 0 3 0 2 7 2 5 5]

#这是第二个，我们看看第一个数据到底是啥样子，woc，是个二位数组，通过print(train_images[0].shape)，我们看到，就是一个28*28的数据，0代表白色，255代表黑色，这样，这个28*28的像素矩阵就构成了一张图，这就是咱们的
# 图啊！
# [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   1   0   0  13  73   0
#     0   1   4   0   0   0   0   1   1   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   3   0  36 136 127  62
#    54   0   0   0   1   3   4   0   0   3]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   6   0 102 204 176 134
#   144 123  23   0   0   0   0  12  10   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0 155 236 207 178
#   107 156 161 109  64  23  77 130  72  15]
#  [  0   0   0   0   0   0   0   0   0   0   0   1   0  69 207 223 218 216
#   216 163 127 121 122 146 141  88 172  66]
#  [  0   0   0   0   0   0   0   0   0   1   1   1   0 200 232 232 233 229
#   223 223 215 213 164 127 123 196 229   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0 183 225 216 223 228
#   235 227 224 222 224 221 223 245 173   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0 193 228 218 213 198
#   180 212 210 211 213 223 220 243 202   0]
#  [  0   0   0   0   0   0   0   0   0   1   3   0  12 219 220 212 218 192
#   169 227 208 218 224 212 226 197 209  52]
#  [  0   0   0   0   0   0   0   0   0   0   6   0  99 244 222 220 218 203
#   198 221 215 213 222 220 245 119 167  56]
#  [  0   0   0   0   0   0   0   0   0   4   0   0  55 236 228 230 228 240
#   232 213 218 223 234 217 217 209  92   0]
#  [  0   0   1   4   6   7   2   0   0   0   0   0 237 226 217 223 222 219
#   222 221 216 223 229 215 218 255  77   0]
#  [  0   3   0   0   0   0   0   0   0  62 145 204 228 207 213 221 218 208
#   211 218 224 223 219 215 224 244 159   0]
#  [  0   0   0   0  18  44  82 107 189 228 220 222 217 226 200 205 211 230
#   224 234 176 188 250 248 233 238 215   0]
#  [  0  57 187 208 224 221 224 208 204 214 208 209 200 159 245 193 206 223
#   255 255 221 234 221 211 220 232 246   0]
#  [  3 202 228 224 221 211 211 214 205 205 205 220 240  80 150 255 229 221
#   188 154 191 210 204 209 222 228 225   0]
#  [ 98 233 198 210 222 229 229 234 249 220 194 215 217 241  65  73 106 117
#   168 219 221 215 217 223 223 224 229  29]
#  [ 75 204 212 204 193 205 211 225 216 185 197 206 198 213 240 195 227 245
#   239 223 218 212 209 222 220 221 230  67]
#  [ 48 203 183 194 213 197 185 190 194 192 202 214 219 221 220 236 225 216
#   199 206 186 181 177 172 181 205 206 115]
#  [  0 122 219 193 179 171 183 196 204 210 213 207 211 210 200 196 194 191
#   195 191 198 192 176 156 167 177 210  92]
#  [  0   0  74 189 212 191 175 172 175 181 185 188 189 188 193 198 204 209
#   210 210 211 188 188 194 192 216 170   0]
#  [  2   0   0   0  66 200 222 237 239 242 246 243 244 221 220 193 191 179
#   182 182 181 176 166 168  99  58   0   0]
#  [  0   0   0   0   0   0   0  40  61  44  72  41  35   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]
#  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
#     0   0   0   0   0   0   0   0   0   0]]

# 第三个，咱们看看有哪些值，哪些分类，你看，一共十个分类
# [0 1 2 3 4 5 6 7 8 9]
#? 具体来说，他们分别代表了'T 恤/上衣', '裤子', '套衫', '连衣裙', '外套', '凉鞋', '衬衫', '运动鞋', '包', '靴子'，很好啊！一切都很OK，应该也很清楚！
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# OK，我们现在来更仔细地看看第一张图，把它画出来！
plt.figure() #创建一个绘图窗口，当然，可以在括号里加入参数figsize=(n, n)来创建窗口大小n*n英寸
plt.imshow(train_images[0]) #Matplotlib会默认将这个二维数组映射为图像。
plt.colorbar() #在图像旁边添加颜色条。需要注意的是，我发现，画出来的图不是黑白，而是彩色，Matplotlib默认使用了一个colormap，叫viridis，我们可以在上面的那个代码中加入cmap='gray'，使得图片变成黑白。
# 我还是继续遵循官网上的代码
plt.grid(False) #默认情况下，会显示网格线，关了之后图像更清晰，现在是关着的
plt.show() #图像，启动！

# 现在，为了训练，我们需要把像素值全部压缩到0-1之间，也就是除以255
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10)) #你看，这里就用了figsize来设置窗口大小，可能因为图有点多吧。
for i in range(25):  #生成0-24数字，然后循环
    plt.subplot(5,5,i+1)  #在这个窗口中，画子图，子图的大小是5*5的网格，也就是容纳25张图！i+1表示当前子图的编号，那第一个就是0+1=1.编号就会让后面画的图依照编号顺序落位。这个i+1必须有啊，不然报错！
    plt.xticks([]) #隐藏x轴的刻度
    plt.yticks([]) #隐藏y轴的刻度
    plt.grid(False) #同之前的一样，关闭网格线
    plt.imshow(train_images[i], cmap=plt.cm.binary) #画图的对象是啥？就是train_images的第i个图。cmap=plt.cm.binary就是灰度模式。和之前的gray有点像，但是好像中间亮度会有点不同，应该都行。
    plt.xlabel(class_names[train_labels[i]]) #为子图的x轴添加标签，取出第i张图的对应标签。train_labels[i]，就会获得train_labels数据集的第i+1个数据，也就是0-9中的一个数字，然后class_names列表通过这个数字
    # 获取对应的名称，作为标签输入。
plt.show()


# 接下来，我们开始构建模型
model = tf.keras.Sequential([  #使用keras的sequential API来构建一个顺序模型，输入数据将会一层一层往下传。
    tf.keras.layers.Flatten(input_shape=(28, 28)),  #第一层，将28*28的二位数组，转化为28*28=784个像素的一维数组，仅仅是重新排列而已。
    tf.keras.layers.Dense(128, activation='relu'),  #全连接层，包括128个神经元，每个神经元都会全量接收上一步的输入内容。使用ReLU激活函数，ReLU意思是，如果输入小于0，那么取0，如果输入大于等于0，那么取输入
    # 可是，我前面输入的像素，必然是0-1之间的数啊，因为在神经元的处理后，可能会出现负值。
    tf.keras.layers.Dense(10)  #输出层，包含10个神经元，对应10个类别，每个神经元输出对应类别的分数，是未归一化的，因此数值可能很大，可能很小，是任意实数。
])
# 额外说一下这个权重矩阵啊，他是什么样的呢？784*128，也就是有784行，128列，以第一行为例，第一行就是128个神经元，给像素1的128种权重，每个神经元都会给他不一样的权重。以第一列为例，那么第一列一共有784个值，也就是
# 第一个神经元给这784个数值都赋予了什么样的权重。最终形成了一个784*128的矩阵。

# 还有这个activation啊，我也有点晕，让我来仔细看看。对单个神经元来说，它是这样的z = w1 * x1 + w2 * x2 + ... + wn * xn + b，用了activation之后，
# 就变成这样了y = activation(z) = activation(w1 * x1 + w2 * x2 + ... + wn * xn + b)，那个偏置项，也是为了让神经元有个基础输出，在输入为0时，也能有个输出。回到activation，在没有activation之前，那
# 就是一个线性方程，z和x1, x2,...的关系都是线性的。activation使它不是线性的了，当然，这个是最简单最快的一种，因为它把负数变为非负的了。

# 编译模型
model.compile(optimizer='adam',  #model.compile是用来配置模型的训练方法的，它指定了下面的三个内容。第一，这一行的optimizer，优化器是为了让模型的预测误差最小，选择的优化器是adam，具体运行方式太难了，看不懂
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #损失函数用来衡量模型预测的结果与真实标签之间的差距，这里用的是SparseCategoricalCrossentropy损失函数，from_logits=True
              #是告诉函数，我给你的是未归一化的，你要先用softmax自己算算。这里还提一点，这个SparseCategoricalCrossentropy适用于整数编码，也就是标签对应的是一个整数，像one-hot编码的向量，就不行
              metrics=['accuracy']) #评估指标，使用accuracy，那就是正确数/总样本数

# 开始输入数据了
model.fit(train_images, train_labels, epochs=10) #epochs是指模型学习数据集的次数，这里要模型学了10次。其他可用函数如batch_size，设置每次训练使用的数据量

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) #评估模型性能，这个verbose函数可以是0，1,2，作用就是0代表啥也不输出，1就是输出进度条和评估结果，2就是只输出评估结果

print('\nTest accuracy:', test_acc) #313/313 - 0s - 821us/step - accuracy: 0.8866 - loss: 0.3357 Test accuracy: 0.8866000175476074

#下面我们可以用训练的模型进行预测
# 创建概率模型，第一层是原来的model，第二层是softmax转化，也就是把未归一化的数值转化为归一化的。
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images) #测试测试集的图，predictions就变成了一个结果合集了
predictions[0] #看看第一个结果如何
np.argmax(predictions[0]) #注意，上面的结果应该是一个包含十个数字的一维数组向量，这里就要找到数值最大的那一个，也就是最可能的那一个。

# 绘制柱状图，显示每个类别的预测概率
def plot_image(i, predictions_array, true_label, img): #i索引，prediction_array就是预测的向量，
  true_label, img = true_label[i], img[i]
  plt.grid(False) #关闭网格线，前面讲过
  plt.xticks([])  #隐藏x轴刻度线，前面讲过
  plt.yticks([])  #隐藏y轴刻度线，前面讲过

  plt.imshow(img, cmap=plt.cm.binary) #灰度图，之前讲过

  predicted_label = np.argmax(predictions_array) #这个前面也有，就是获取向量中数值最大的那个数字
  if predicted_label == true_label: #这里用蓝色和红色表示是不是预测对了
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],  #模型预测的标签
# 专门讲一下这个"{} {:2.0f}% ({})"，应当说是用来控制输出的格式，分为三个部分，{}和{:2.0f}% 和({})，第一个{}就是一个占位符，这里可以是任何东西，后续这里会是服饰的类别名称，真实的类别名称，第二个同样先通过{}
# 创建了一个占位符，然后，里面放入冒号:，告诉你，开始说明这个里面的格式是什么了，2表示，宽度是两个字符，.0f表示保留小数点后0位，f为浮点数的意思，%表示一个百分号，因为预测是概率，百分之多少是什么样的一个东西。
# 第三个则是({})，表示内容会被放到括号里，括号里将会是模型认为的最可能的东西。
                                100*np.max(predictions_array), #显示相应的预测概率
                                class_names[true_label]), #获取真实的标签分类，用来和上面的模型预测标签作比较。
                                color=color) #设置标签的颜色。

def plot_value_array(i, predictions_array, true_label): #同上，i索引， prediction_array预测概率分布，一个向量。
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777") #颜色默认灰色，奇怪的一点是为什么会有thisplot这个东西，事实上通过thisplot这一段代码的标记，我们获得了十个柱子的控制，后续可以对其颜色
#   进行修改
  plt.ylim([0, 1]) #y轴范围0-1，因为前面用softmax进行了归一化
  predicted_label = np.argmax(predictions_array) 

  thisplot[predicted_label].set_color('red') #显示模型对每个类别的预测概率
  thisplot[true_label].set_color('blue') 


num_rows = 5  #和之前展示训练集是一样的，创建五行三列，画十五个图
num_cols = 3
num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows)) #这里就是设置画布的大小，可以看到，每一列会有三个图，想给2个网格线，但是每一列除了服饰图像本身，还会有他对应的一个概率柱状图，所以再乘以2.行就是直接给
# 两个网格线，那就直接乘2.

for i in range(num_images): #和之前一样，开始创建子画布，这里遍历后，分别绘制服饰图和概率柱状图
  plt.subplot(num_rows, 2*num_cols, 2*i+1) #行就是前面的，行数，列则如我前面所说，因为有概率柱状图，乘以2了，然后，编号，必不可少。
  plot_image(i, predictions[i], test_labels, test_images) #然后，这里，画图
  plt.subplot(num_rows, 2*num_cols, 2*i+2) #这里可以看到，编号就是让它和前面错开，从而一一对应。
  plot_value_array(i, predictions[i], test_labels) 
plt.tight_layout() #自动调整空白，使得画布紧凑
plt.show()

img = test_images[1] #试试第一张图

img = (np.expand_dims(img,0)) #要把这个二维数组变成三维的？

predictions_single = probability_model.predict(img)

print(predictions_single)


plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45) #rotation旋转45度，使得标签便于阅读
plt.show()

np.argmax(predictions_single[0])



# 现在，我们来做一个我自己的内容，逆水寒手游中有很多内功，但是他们属于金木水火土五个分类之一，也有内功融合了两种元素，我想试试如果通过外观，能不能让机器识别出这个内功是什么元素
import os
from PIL import Image
image_library = "C:/Users/64171/Desktop/Elements" #这是我的目录
#下面是我将不同内功放入不同的子文件夹
categories = ['金','木','水','火','土']

# 先创建空列表，我们将内功数据库逐渐加入进去
data = []
labels = []

for label, category in enumerate(categories): #现在，我们就要开始遍历每个文件夹了.注意，这个enumerate函数，会自动给返回的结果添加索引，也就是每遍历一个文件夹，自动给他一个编号，label自动添加上去了。
    #这样，每一个图片，它在输出的时候，都会被贴上对应的文件夹的label
    file_path = os.path.join(image_library, category) #我们现将路径进行合并。拼出每个分类的子文件路径。
    for image in os.listdir(file_path):
       image_path = os.path.join(file_path, image) #进一步拼接，拼出每个图片的文件路径。
       image = Image.open(image_path).convert('L')   #打开图，并且转化为灰度图。
       image_array = np.array(image) #同样，转化为数组
       data.append(image_array) #将转化后的这个二维数组，加入到数据集中
       labels.append(label)


data = np.array(data)/255 #和前面一样，将0-255的像素归到0-1区间
labels = np.array(labels) #将这个列表转化为一个一维数组
print(data.shape)
print(labels.shape)

#(34, 138, 138)
# (34,)
# 上面是结果啊，还是成功的，34张图。下面我们来构建模型。

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(138, 138)),  # 输入的图片是138*138，所以这里改成138
    tf.keras.layers.Dense(128, activation='relu'),    # 不变！
    tf.keras.layers.Dense(5)    # 我们只有金木水火土五个类别，所以改成5个！
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping( #这个函数非常好懂，里面的参数都是很标准的英语。
    monitor='val_loss',  # 监控损失这一指标
    patience=5,          # 如果验证集损失连续 5 轮没有提升，停止训练
    restore_best_weights=True  # 恢复到验证集性能最好的权重
)
model.fit(data, labels, epochs=50, callbacks=[early_stopping])

# 在这里，我发现了一个问题，我预先设置的是训练十轮，但是结果告诉我：
# “Epoch 1/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 24ms/step - accuracy: 0.1801 - loss: 2.8419 
# Epoch 2/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.2102 - loss: 14.5704
# Epoch 3/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.3303 - loss: 6.6725
# Epoch 4/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.1697 - loss: 9.9408 
# Epoch 5/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1501 - loss: 5.7031
# Epoch 6/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.2598 - loss: 2.9296
# Epoch 7/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1998 - loss: 3.0455
# Epoch 8/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.2102 - loss: 3.4229
# Epoch 9/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.2102 - loss: 1.9663
# Epoch 10/10
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.3094 - loss: 2.3802”
# 可以看到，准确率不断波动，于是我提高了训练轮次，如下：
# “Epoch 1/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step - accuracy: 0.1201 - loss: 3.1592 
# Epoch 2/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3199 - loss: 22.0484
# Epoch 3/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.3603 - loss: 17.8683
# Epoch 4/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1998 - loss: 12.2298
# Epoch 5/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.3002 - loss: 3.1205
# Epoch 6/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.2598 - loss: 3.9203
# Epoch 7/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1893 - loss: 5.3345
# Epoch 8/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.2702 - loss: 4.7120
# Epoch 9/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.1801 - loss: 2.6470
# Epoch 10/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.1998 - loss: 3.5455
# Epoch 11/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.4504 - loss: 1.7801
# Epoch 12/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.2402 - loss: 2.2252
# Epoch 13/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.5404 - loss: 1.4328
# Epoch 14/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.2898 - loss: 2.0863
# Epoch 15/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.4295 - loss: 1.4313
# Epoch 16/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3002 - loss: 1.2737
# Epoch 17/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.2598 - loss: 1.5952
# Epoch 18/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.4099 - loss: 1.2606
# Epoch 19/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.2898 - loss: 1.3698
# Epoch 20/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.4203 - loss: 1.5257
# Epoch 21/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3499 - loss: 1.2937
# Epoch 22/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.4203 - loss: 1.3803
# Epoch 23/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3799 - loss: 1.6801
# Epoch 24/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.5196 - loss: 1.1974
# Epoch 25/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.3002 - loss: 1.5174
# Epoch 26/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.5600 - loss: 0.9484
# Epoch 27/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.3499 - loss: 1.2916
# Epoch 28/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.1998 - loss: 1.6019
# Epoch 29/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.6397 - loss: 1.0284
# Epoch 30/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.6906 - loss: 0.9794
# Epoch 31/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.4596 - loss: 1.1767
# Epoch 32/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.6501 - loss: 0.8682
# Epoch 33/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.6605 - loss: 0.8500
# Epoch 34/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.6501 - loss: 0.9618
# Epoch 35/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.7298 - loss: 0.9010
# Epoch 36/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step - accuracy: 0.5300 - loss: 0.9587
# Epoch 37/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.5000 - loss: 1.0649
# Epoch 38/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.7298 - loss: 0.7586
# Epoch 39/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.5901 - loss: 1.0501
# Epoch 40/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step - accuracy: 0.4099 - loss: 1.3673
# Epoch 41/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.4700 - loss: 1.1719
# Epoch 42/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 14ms/step - accuracy: 0.6097 - loss: 0.7892
# Epoch 43/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.7702 - loss: 0.7647
# Epoch 44/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.4896 - loss: 1.1093
# Epoch 45/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.7598 - loss: 0.7785
# Epoch 46/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.6998 - loss: 0.7460
# Epoch 47/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.4596 - loss: 1.1263
# Epoch 48/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 17ms/step - accuracy: 0.4203 - loss: 1.0210
# Epoch 49/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 16ms/step - accuracy: 0.7506 - loss: 0.7264
# Epoch 50/50
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.5600 - loss: 1.0391”
# 也在不断地反复啊，所以，模型训练可能存在一些问题，第一，我提供的所谓内功，本就可能拥有较少的共性，第二，训练轮次加大，虽然拟合程度提高了，但是也可能出现过拟合的问题，因此，我需要引用一个“早停”机制
#? from tensorflow.keras.callbacks import EarlyStopping

#? early_stopping = EarlyStopping( #这个函数非常好懂，里面的参数都是很标准的英语。
#?     monitor='val_loss',  # 监控损失这一指标
#?     patience=5,          # 如果验证集损失连续 5 轮没有提升，停止训练
#?     restore_best_weights=True  # 恢复到验证集性能最好的权重
#? )
#?当然，上面的内容我就直接加到前面去了。

# OK牛逼，老子跑了100轮，他直接给我把准确率干到100去了！
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 1.0000 - loss: 0.2569

# 接下来，我们用完全一样的步骤，创建我们的测试集

test_library = "C:/Users/64171/Desktop/ETEST"  
categories = ['金', '木', '水', '火', '土']  

test_data = []
test_labels = []


for label, category in enumerate(categories):
    file_path = os.path.join(test_library, category)  
    for image in os.listdir(file_path):
        image_path = os.path.join(file_path, image)  
        image = Image.open(image_path).convert('L')  
        image_array = np.array(image)  
        test_data.append(image_array)  
        test_labels.append(label)  

test_data = np.array(test_data)/255.0  
test_labels = np.array(test_labels)  

print(test_data.shape)  
print(test_labels.shape)  

# 然后，我们开始构建概率模型
label_dictionary = {label: category for label, category in enumerate(categories)}
print(label_dictionary)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])#前面真的完全一样
predictions = probability_model.predict(test_data)
print(predictions[0]) 
np.argmax(predictions[0])
print(np.unique(test_labels))
# 到这里才发现，我虽然对应了数字，但是竟然不知道究竟是哪个分类，所以我在前面加个这个label_dictionary = {label: category for label, category in enumerate(categories)}

# 接下来，我将进行测试，我用了另外13个测试内功来进行测试，而代码则是和前面的完完全全一毛一样！
class_names = ['金', '木', '水', '火', '土']

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)  

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'  
    else:
        color = 'red'  

    plt.xlabel("{} {:2.0f}% ({})".format(
        class_names[predicted_label],  
        100 * np.max(predictions_array),  
        class_names[true_label] 
    ), color=color)  


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(len(class_names)), class_names, rotation=45)  
    plt.yticks([])
    thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])  

    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')  
    thisplot[true_label].set_color('blue')  



num_rows = 5
num_cols = 3
num_images = 13 #所有内容都没变，就这里变了，因为我只有13个用来测试的内功！

plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_data)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 好的，最终结果也跑出来了，成绩惨不忍睹啊，正确率差不多只有一半，不过，我发现，金内功和土内功一共五个，都
# 预测对了，而木水火内功则几乎全错，所以我觉得，一定程度上，可能金内功和土内功的内部共享很多相近的元素，比如
# 土内功的山岳，而金内功则是刀剑利器！我会把最终结果拿出来放到网上。

#更正一下啊，我尝试了学习的不同次数，发现土内功也没那么稳定，金内功是真的几乎一直都正确！
