# 其实只是都出来了whole list


#循环读取文件夹里的list，并且根据undergraduate里的school分类存储
j=0
#读取文件
wholeList = []

import json
for i in range(1, 300):
    with open('list'+str(i)+'.json','r',encoding='utf8') as f:
        offerContent = json.load(f)
        for offer in offerContent["data"]["data"]:
            wholeList.append(offer)
            j=j+1
            print(j)
            print(offer["schoolname"])
            #根据学校名字分类

#写wholelist
with open('wholeList.json','w',encoding='utf8') as f:   
    json.dump(wholeList, f, ensure_ascii=False)

print(len(wholeList))