import cv2   
import easyocr
import re
import pandas as pd

# Now, we'll start reading the image and use OpenCV to crop the segments we need.
# Previously, I tried to recognize the entire attribute value area as a whole, but the results were poor.
# Therefore, I decided to completely segment the attribute area into four regions with almost no interference:
# "Health Points, Basic Attributes, Combat Attribute 1, Combat Attribute 2"
image = cv2.imread("C:/Users/64171/Desktop/folder1/picture4.png")
image1 = image[566:680, 1807:2351]
# I also tried using the following line of code to directly view the cropped image,
# but the window opens and closes instantly. 
# Here's the correct way to solve it:
# cv2.imshow("image1", image1)
# cv2.waitKey(0)  # The window will wait for the user to press any key before closing
# cv2.destroyAllWindows()

# I also tried using the following function, which pops up a window allowing you to select the area manually.
# However, since all my screenshots will have the same format, I can just specify the area in the code.
# Additionally, I don't want to manually select the area every time the code runs.
# r = cv2.selectROI("select the area", image)
# You can also use the following code to save the cropped image and check if it's the region you want.
# cv2.imwrite("image1.png", image1)

# Now let's repeat the same cropping for the other regions as well.
image2 = image[675:875, 1807:2422]
image3 = image[861:1110, 1807:2422]
image4 = image[1112:1361, 1807:2422]

# cv2.imwrite("image2.png", image2)
# cv2.imwrite("image3.png", image3)
# cv2.imwrite("image4.png", image4)

# Next, we use easyocr to extract text from the images. It seems to be very powerful.
# I tried using Tesseract, but the results were unsatisfactory.
# Additionally, easyocr performs well with Chinese, so I chose to use easyocr for this task.
# Use the following function to set the desired language. Since it's a Chinese game, I set it to Chinese.
Reader = easyocr.Reader(["ch_sim"]) 

# Then, use easyocr to directly read the text from these regions.
text1 = Reader.readtext(image1)
text2 = Reader.readtext(image2)
text3 = Reader.readtext(image3)
text4 = Reader.readtext(image4)

# The result will be a list with multiple sub-lists. Each sub-list contains the coordinates of the recognized text, 
# the text itself, and the confidence level. However, we only need the extracted text, 
# so we use the following function to loop through each region and retrieve the text from each sub-list.
text_1 = [text[1] for text in text1]
text_2 = [text[1] for text in text2]
text_3 = [text[1] for text in text3]
text_4 = [text[1] for text in text4]

# Of course, we can use the print function to check the results.
# However, during this process, we encountered some issues. 
# Due to the proximity of certain numbers and text, the recognized content is "stuck together."
# For example, in region 2, we can see that '594' and 'Strength' are stuck together.
# But in reality, '594' is the value corresponding to the 'Vitality' attribute.
# print(text_1) 
# print(text_2)  # ['Vitality', '594  Strength', '89', 'Constitution', '361', 'Agility', '486', 'Endurance', '390']
# print(text_3)
# print(text_4)

# Therefore, I want to create a function to separate the attributes and values.
# The following code uses regular expressions to split all attributes and values.
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

# However, the above code incorrectly splits the value for 'Health Points', so a slight modification is needed:
def qiege(list):
    list0 = []

    for item in list:
        # Add a condition here
        if "/" in item:
            list0.append(item)
        else:
            split_items = re.findall(r'[^\d\s]+|\d+', item)
            # Let me explain the regular expression. First, re.findall() is a function that takes a pattern and a string.
            # The pattern here is a combination of characters and digits. The 'r' before the pattern ensures that 
            # Python doesn't treat the backslash as an escape character. The square brackets [] define a character class,
            # meaning it will match any character inside the brackets. \d matches a digit character, and \s matches a 
            # whitespace character. The ^ at the beginning negates these, meaning it will match anything that is 
            # neither a number nor a whitespace, and in this case, Chinese characters fall into this category.
            # The + means the match will occur multiple times. The vertical bar | means "or," so it will match either 
            # the left or right expression. In this list, I will extract both Chinese characters and numbers.
            list0.extend(split_items)
            # We use 'extend' instead of 'append' because 'extend' adds individual elements, not sublists. 
            # For example, when encountering the element '594 Strength' and splitting it into ['594', 'Strength'], 
            # 'extend' adds each element individually, whereas 'append' would add the entire sublist.

    return list0

lista = qiege(text_1)
print(lista)
listb = qiege(text_2)
print(listb)
listc = qiege(text_3)
print(listc)
listd = qiege(text_4)
print(listd)

# Now it works. Next, we need to pair the attributes with their corresponding values, 
# in the form of "attribute - value."

def pair_attributes_and_values(list):
    # As usual, let's start with an empty list
    final_list = []

    for i in range(0, len(list), 2):
        # By setting the step size to 2, we can get only the attribute names.
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

# Now, let's make the output look a bit nicer.
for element in shuxing:
    print(element)
        
# Next, let's try to save this content to an Excel file.
attributes = []
values = []

for element in shuxing:
    attribute, value = element.split(": ")
    attributes.append(attribute)
    values.append(value)

df = pd.DataFrame({"Attribute": attributes, "Value": values})

df.to_excel("attributes_values.xlsx", index = False)

# Success!