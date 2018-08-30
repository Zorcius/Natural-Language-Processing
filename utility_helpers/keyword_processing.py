import numpy as np

file = open("C:\\Zorcius\\filtering\\keyword_new0927.txt",'r',encoding='utf-8')
out = open("C:\\Zorcius\\filtering\\keyword_new.txt",'w',encoding='utf-8')
keyword = []
lines = []

def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
        return False

for line in file.readlines():
    if line != '\n':
        lines.append(line.split('|'))

for index in range(len(lines)):
    for lst in lines[index]:
        out.write(lst+'\n')
out.close()