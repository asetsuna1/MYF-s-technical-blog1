import cv2   
import easyocr
import re
import pandas as pd

# 现在，我们开始读取图片，并且使用OpenCV切取我们所需要的片段。此前，我尝试将整个属性数值区域作为一个整体进行识别，但是效果不佳，因此我尝试将属性区域完全切开，分成几乎没有任何干扰元素的四个区域，分别是“气血、基础、战斗属性1、战斗属性2”
image = cv2.imread("C:/Users/64171/Desktop/folder1/picture4.png")
image1 = image[566:680, 1807:2351]
# cv2.imshow("image1", image1)我也尝试使用这一行代码来直接查看切取图片，但是窗口会秒打开秒关闭，有没有懂哥教教我。
# 找到方法了啊兄弟们，方法如下：
# cv2.imshow("image1", image1)
# cv2.waitKey(0) 用户按任意键，窗口才会关闭
# cv2.destroyAllWindows()

# r = cv2.selectROI("select the area", image) 我也尝试过这个函数，它会弹出一个窗口，让你直接选择你所需要的区域，不过，一方面，我的所有截图格式将会完全一样，我只需要使用代码指定区域即可，另一方面，我也不想每次运行代码都手动框选一个区域。
# cv2.imwrite("image1.png", image1)可以使用这样的代码来将切片保存，看看是否切取了我所需要的片段。

# 下面就是重复此前的全部操作了。
image2 = image[675:875, 1807:2422]
image3 = image[861:1110, 1807:2422]
image4 = image[1112:1361, 1807:2422]

# cv2.imwrite("image2.png", image2)
# cv2.imwrite("image3.png", image3)
# cv2.imwrite("image4.png", image4)

# 接下来，使用easyocr来读取图片中的内容。其功能感觉非常强大。我尝试了使用tesseract，但是效果不佳。
# 此外，eassyocr在处理中文时表现不错，因此我尝试使用了easyocr
# 使用下面这个函数，可以设置你所需要的语言，中文游戏自然我就设置了中文。
Reader = easyocr.Reader(["ch_sim"]) 

# 接着，使用easyocr直接读取这些区域。
text1 = Reader.readtext(image1)
text2 = Reader.readtext(image2)
text3 = Reader.readtext(image3)
text4 = Reader.readtext(image4)

# 由于读取出来的结果将会是一个列表和多个子列表，每个子列表式识别的文字所在坐标、文本内容、以及可信度。但是，我们只需要获得具体文本是什么就行了，因此使用函数，遍历这四个区域，也就是四个列表，读取每个子列表的文本内容，就OK了。
text_1 = [text[1] for text in text1]
text_2 = [text[1] for text in text2]
text_3 = [text[1] for text in text3]
text_4 = [text[1] for text in text4]

# 当然，使用print函数看看效果，不过在这个过程中，我们就发现了一些问题，由于部分数值和文本距离较近，识别出来的内容出现了“黏连”
# print(text_1) 
# print(text_2) ['气海', '594  力量', '89', '根骨', '361', '身法', '486', '耐力', '390'] 以区域二为例，我们可以看到，594数值和力量黏连到了一起，但是他实际上是气海这个属性对应的数值。
# print(text_3)
# print(text_4)

# 因此，我希望通过一个函数，首先将这些属性和数值，切割开
# 一下代码使用了正则表达式，通过这个函数，我将所有的属性和数值都进行了切割。
# def qiege(list):
#     list0 = []

#     for item in list:
#         split_items = re.findall(r'[^\d\s]+|\d+', item)
#         list0.extend(split_items)

#     return list0

# lista = qiege(text_1)
# print(lista)
# listb = qiege(text_2)
# print(listb)
# listc = qiege(text_3)
# print(listc)
# listd = qiege(text_4)
# print(listd)

# 不过，上面的代码把气血的数值给切爆了，所以需要略微修改，如下：
def qiege(list):
    list0 = []

    for item in list:
        # 搞个判断进来
        if "/" in item:
            list0.append(item)
        else:
            split_items = re.findall(r'[^\d\s]+|\d+', item)
            # 让我来解释一下这个正则表达式。首先，re.findall()函数，re.findall(pattern, string, flags = 0), pattern是你想要的匹配形式，我这里是文字+数字。string则是应用的字符串，在pattern中的细节是，r''保证python不会吧\理解为转义符，不是用来换行的。[]表示字符类，也就是说，用来匹配方括号内的任意字符。\d表示匹配数字字符，\s表示匹配空白字符，而在前面使用^，则是取反义，意味着匹配既不是数字也不是空白的字符，而中文在这里就是一个非数字，非空白的字符。+代表着，这样的匹配将会进行多次。在这之后，|表示或者，也就是匹配左边的表达式，或者右边的表达式，也就是说，在这个列表中，你只要是汉字或者数字，我都会把你选择出来。
            list0.extend(split_items)
            # 这里使用了extend而不是append，因为这一步之后，extend是将一个个元素加入，而不会将列表加入。比如我遍历的前面出现的“594 力量”这个元素时，通过函数，我将其切割为了["594", "力量"]，如果使用append，就会将这个列表直接作为单独元素加入到最终列表中，而使用extend则会将这个子列表中的元素逐个作为元素加入最终列表。

    return list0

lista = qiege(text_1)
print(lista)
listb = qiege(text_2)
print(listb)
listc = qiege(text_3)
print(listc)
listd = qiege(text_4)
print(listd)

# 这下就OK了啊。接下来，我们需要把这些属性开始配对了，属性-数值这样的形式进行。

def pair_attributes_and_values(list):
    # 老规矩，先搞个空列表
    final_list = []

    for i in range(0, len(list), 2):
        # 通过设置步长为2，可以只取属性名称。
        attributes = list[i]
        value = list[i+1]
        final_list.append(f"{attributes}: {value}")
    
    return final_list

final1 = pair_attributes_and_values(lista)
final2 = pair_attributes_and_values(listb)
final3 = pair_attributes_and_values(listc)
final4 = pair_attributes_and_values(listd)

# print(final1)
# print(final2)
# print(final3)
# print(final4)

shuxing = final1 + final2 + final3 + final4
# print(shuxing)

# 接下来，我们让这个输出结果好看一点。
for element in shuxing:
    print(element)
        
#接下来，我们试着把这些内容保存到Excel表格中。
attributes = []
values = []

for element in shuxing:
    attribute, value = element.split(": ")
    attributes.append(attribute)
    values.append(value)

df = pd.DataFrame({"属性": attributes, "数值": values})

df.to_excel("属性数值表.xlsx", index = False)

# 成功了啊嗯




