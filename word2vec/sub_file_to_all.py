#coding=utf-8
import os
path = os.getcwd()
filedir = path[:-9] + "\\aclImdb\\train\\unsup"  #获取目标文件夹的路径
#获取当前文件夹中的文件名称列表
filenames=os.listdir(filedir)
#打开当前目录下的result.txt文件，如果没有则创建
f=open(path[:-9] + '\\aclImdb\\train\\unsup_all.txt', 'w', encoding="utf-8")
#先遍历文件名
for filename in filenames:
    filepath = filedir+'/'+filename
    #遍历单个文件，读取行数
    for line in open(filepath, encoding="utf-8"):
        f.writelines(line)
    f.write('\n')
#关闭文件
f.close()
