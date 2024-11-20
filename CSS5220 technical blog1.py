import cv2   
import easyocr
import re
import pandas as pd

# Now, let's begin by reading the image and using OpenCV to extract the segments we need. 
# Previously, I tried to recognize the entire attribute value region as a whole, 
# but the results were not satisfactory. Therefore, I tried to completely separate the attribute region,
# dividing it into four areas with almost no interference: "Vitality, Base Attributes, Combat Attributes 1, Combat Attributes 2".
image = cv2.imread("C:/Users/64171/Desktop/folder1/picture4.png")
image1 = image[566:680, 1807:2351]
# cv2.imshow("image1", image1) I also tried using this line of code to directly view the cropped image,
# but the window opens and closes immediately. Is there anyone who can teach me how to fix this?
# Found the solution, brothers! The method is as follows:
# cv2.imshow("image1", image1)
# cv2.waitKey(0) Let the user press any key, then the window will close.
# cv2.destroyAllWindows()

# r = cv2.selectROI("select the area", image) I also tried this function. It will pop up a window
# allowing you to directly select the area you need. However, on one hand, all my screenshots will have the exact same format, 
# so I only need to specify the region using code. On the other hand, I don't want to manually select a region every time I run the code.
# cv2.imwrite("image1.png", image1) You can use this line of code to save the cropped image and check if it captures the part you need.

# Now, repeat the same operations as before.
image2 = image[675:875, 1807:2422]
image3 = image[861:1110, 1807:2422]
image4 = image[1112:1361, 1807:2422]

# cv2.imwrite("image2.png", image2)
# cv2.imwrite("image3.png", image3)
# cv2.imwrite("image4.png", image4)

# Next, use easyocr to read the text from these images. It seems to be a very powerful tool.
# I tried using tesseract, but it didn't work well. Also, easyocr performs well when handling Chinese text, 
# which is why I chose to use easyocr. The following function allows you to set the language you need. 
# Since it's a Chinese game, I set it to Chinese.
Reader = easyocr.Reader(["ch_sim"]) 

# Then, use easyocr to directly read these regions.
text1 = Reader.readtext(image1)
text2 = Reader.readtext(image2)
text3 = Reader.readtext(image3)
text4 = Reader.readtext(image4)

# Since the results will be a list containing multiple sub-lists, where each sub-list consists of the coordinates of the recognized text,
# the content of the text, and the confidence level, we only need to extract the actual text.
# To do this, I wrote a function to iterate over these four regions (i.e., four lists) 
# and extract the text content from each sub-list.
text_1 = [text[1] for text in text1]
text_2 = [text[1] for text in text2]
text_3 = [text[1] for text in text3]
text_4 = [text[1] for text in text4]

# Of course, you can use the print function to check the results. During this process, 
# I noticed some issues. Since some values are too close to the text, the recognized content is "stuck together."
# print(text_1) 
# print(text_2) ['Vitality', '594Strength', '89', 'Constitution', '361', 'Agility', '486', 'Endurance', '390']
# For example, in region two, the value 594 is stuck with the "Strength" attribute, 
# but in reality, it's the value corresponding to the "Vitality" attribute.
# print(text_3)
# print(text_4)

# Therefore, I hope to use a function to split the attributes and values first.
# The following code uses regular expressions. With this function, 
# I split all the attributes and values.
# def qiege(list):
#     list0 = []

#     for item in the list:
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

# However, the above code splits the Vitality value incorrectly, 
# so a slight modification is needed, as follows:
def qiege(list):
    list0 = []

    for item in list:
        # Add a condition here
        if "/" in item:
            list0.append(item)
        else:
            split_items = re.findall(r'[^\d\s]+|\d+', item)
            # Let me explain this regular expression. First, the re.findall() function, 
            # re.findall(pattern, string, flags = 0), where pattern is the specific matching pattern you want. 
            # I am matching text + numbers here. The string is the string to apply the regular expression on.
            # The key details in pattern are: r'' ensures Python doesn't treat the backslash as an escape character.
            # [] represents a character class, which matches any character inside the brackets. 
            # \d matches digits, \s matches whitespace, and ^ at the front negates the class, meaning it matches characters 
            # that are neither digits nor whitespace. Chinese characters fall into this category.
            # The + means the match will occur one or more times. After that, | means "or", 
            # which means matching either the expression on the left or on the right.
            # In this list, if it's either a Chinese character or a number, I will extract it.
            list0.extend(split_items)
            # I used extend instead of append here because extend adds elements one by one, 
            # whereas append would add the entire list as a single element. 
            # For example, when processing the item "594Strength", I split it into ["594", "Strength"].
            # If I used append, it would add the list as a whole, but using extend will add each element individually.

    return list0

lista = qiege(text_1)
print(lista)
listb = qiege(text_2)
print(listb)
listc = qiege(text_3)
print(listc)
listd = qiege(text_4)
print(listd)

# Now it's OK. Next, we need to pair these attributes with their corresponding values, 
# i.e., in the form "attribute - value".

def pair_attributes_and_values(list):
    # As usual, start with an empty list
    final_list = []

    for i in range(0, len(list), 2):
        # By setting the step size to 2, we can extract only the attribute names.
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

# Next, let's make the output look better.
for element in shuxing:
    print(element)
        
# Next, let's try saving this content to an Excel file.
attributes = []
values = []

for element in shuxing:
    attribute, value = element.split(": ")
    attributes.append(attribute)
    values.append(value)

df = pd.DataFrame({"Attribute": attributes, "Value": values})

df.to_excel("AttributeValueTable.xlsx", index = False)

# It worked! 
import cv2
import easyocr
import re
import pandas as pd

images = cv2.imread("C:/Users/64171/Desktop/folder1/picture5.png")

image11 = images[348:1386, 1485:1949]
image22 = images[348:1306, 1992:2454]

# cv2.imshow("image2", image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows

Reader = easyocr.Reader(["ch_sim"])

text11 = Reader.readtext(image11)
text22 = Reader.readtext(image22)

text_11 = [text[1] for text in text11]
text_22 = [text[1] for text in text22]

def qiege2(list):
    list0 = []

    for i in list:
        if any(char in i for char in ["/", "-", "."]):
            list0.append(i)

        else:
            split_items = re.findall(r'[^\d\s]+|\d+', i)

            list0.extend(split_items)
    
    return list0

lista2 = qiege2(text_11)
listb2 = qiege2(text_22)

# print(lista)
# print(listb)
# Here I noticed that some 0 values were not recognized. I wanted to add 0 to the list through detection. 
# Unfortunately, this step was unsuccessful.
# def add_zeros(list):

#     list_s = []

#     for i in range(0, len(list)-1):
#         list_s.append(list[i])
#         if list[i].isdigit() == False and list[i+1].isdigit() == False:
#             list_s.append('0')
    
#     list_s.append(list[-1])
#     return list_s

# lista = add_zeros(lista)
# listb = add_zeros(listb)
# print(lista)
# print(listb)
def is_number(i):
    try:
        # Check if it is a float, then use replace to remove the various symbols 
        # that might interfere with the judgment.
        float(i.replace("/", "").replace("-", "").replace(".", ""))
        return True
    except ValueError:
        return False


def add_zeros(list):
     
    list_s = []

    for i in range(len(list)-1):
        list_s.append(list[i])
        if not is_number(list[i]) and not is_number(list[i+1]):
            list_s.append('0')
    
    list_s.append(list[-1])

    if not is_number(list[-1]):
        list_s.append('0')
    return list_s

listaa = add_zeros(lista2)
listbb = add_zeros(listb2)
# print(listaa)
# print(listbb)

# The remaining operations are the same.
def pair_attributes_and_values2(list):
    final = []
    for i in range(0, len(list), 2):
        attribute = list[i]
        value = list[i+1]
        final.append(f"{attribute}:{value}")
    return final
finala = pair_attributes_and_values2(listaa)
finalb = pair_attributes_and_values2(listbb)

final_list2 = finala + finalb

attributes2 = []
values2 = []
for elements in final_list2:
    attribute, value = elements.split(":")
    attributes2.append(attribute)
    values2.append(value)

df2 = pd.DataFrame({"Attribute": attributes2, "Value": values2})
df2.to_excel("AttributeValueTable2.xlsx", index = False)