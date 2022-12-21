#读 agreeList 转成string 存到 offer_string_list
import json

with open('agreeList.json','r',encoding='utf8') as f:
    agreeList = json.load(f)


offer_string_list=[]
for offer in agreeList:
    offer_string=""

    # if "gre" in offer["userinformation"]:
    #     if "total" in offer["userinformation"]["gre"]:
    #         offer_string += str(offer["userinformation"]["gre"]["total"])
    #         print(str(offer["userinformation"]["gre"]["total"]))
    #         print(json.dumps(offer,ensure_ascii=False))
    #     else:
    #         offer_string += "300"
    offer_string+=json.dumps(offer["userinformation"],ensure_ascii=False)
    offer_string_list.append({"offer_string":offer_string,"school":offer["schoolname"]})

    #查看userinformation有没有ielts子目录
    # if "ielts" in offer["userinformation"]:
    #     ielts+=1
    # #查看userinformation有没有tofel子目录
    # if "toefl" in offer["userinformation"]:
    #     tofel+=1
    # #查看userinformation有没有undergraduate子目录
    # if "undergraduate" in offer["userinformation"]:
    #     undergraduate+=1

with open('offer_string_list.json','w',encoding='utf8') as f:
    json.dump(offer_string_list,f,ensure_ascii=False)

