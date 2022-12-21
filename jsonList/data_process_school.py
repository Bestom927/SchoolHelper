#读 agreelist 再根据school分类 写到schoolList里
import json

with open('agreeList.json','r',encoding='utf8') as f:
    wholeList = json.load(f)

schoolList = []
schoolNumList=[]
agreeList = []
rejectList = []
gre=0
ielts=0
tofel=0
undergraduate=0
j=0
#循环读取文件夹里的list，并且根据undergraduate里的school分类存储
for offer in wholeList:
    #print(offer["schoolname"])
    #根据学校名字分类
    if offer["schoolname"] not in schoolList:
        schoolNumList.append({"shcool":offer["schoolname"],"num":1})
        schoolList.append(offer["schoolname"])
        # with open('schoolList.json','w',encoding='utf8') as f:   
        #     json.dump(schoolList, f, ensure_ascii=False)
    else:
        for i in range(len(schoolNumList)):
            if schoolNumList[i]["shcool"]==offer["schoolname"]:
                schoolNumList[i]["num"]+=1
        j=j+1

    #如果录取了
    if offer["apply_resultstatus"]==4:
        rejectList.append(offer)
        # with open('agreeList.json','w',encoding='utf8') as f:   
        #     json.dump(agreeList, f, ensure_ascii=False)
    #如果拒绝了
    else:
        agreeList.append(offer)
        # with open('rejectList.json','w',encoding='utf8') as f:   
        #     json.dump(rejectList, f, ensure_ascii=False)
    #查看userinformation有没有gre子目录
    if "gre" in offer["userinformation"]:
        gre+=1
    #查看userinformation有没有ielts子目录
    if "ielts" in offer["userinformation"]:
        ielts+=1
    #查看userinformation有没有tofel子目录
    if "toefl" in offer["userinformation"]:
        tofel+=1
    #查看userinformation有没有undergraduate子目录
    if "undergraduate" in offer["userinformation"]:
        undergraduate+=1
  

# with open('agreeList.json','w',encoding='utf8') as f:   
#     json.dump(agreeList, f, ensure_ascii=False)
# with open('rejectList.json','w',encoding='utf8') as f:   
#     json.dump(rejectList, f, ensure_ascii=False)

#存school list
with open('schoolList.json','w',encoding='utf8') as f:
    json.dump(schoolList, f, ensure_ascii=False)

for school in schoolList:
    print(school)

print(len(schoolList))
print(len(agreeList))
print(len(rejectList))
print(j)
# print(schoolNumList)
# #print(schoolList)

# print("gre:",gre)
# print("ilets:",ielts)
# print("tofel",tofel)
# print("undergraduate:",undergraduate)