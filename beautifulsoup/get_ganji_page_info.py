#-*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from util import mongodbHelper
headers = {
    'user-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.94 Safari/537.36'
}
home_url = 'http://sz.ganji.com/xixiangsz/zufang/0/'
wb_data = requests.get(home_url,headers = headers)
soup = BeautifulSoup(wb_data.text,'html.parser')
utag = soup.select('div.f-list-item.ershoufang-list')
db = mongodbHelper.mongoDB()
db.delete_all("ganjicol")
for item in utag:
    price = ""
    address_info = ""
    size_info = ""
    title = ""
    img_path = ""
    time_info = ""
    print(item.get_text)
    price_num = item.select("span.num")[0].text.strip()
    price_tag = item.select("span.yue")[0].text.strip()
    address_eara = item.select("span.address-eara")
    img_info = item.select("img")
    size_span = item.select("dd.size > span")
    time_info = item.select("div.time")[0].text.strip()
    href_info = "http:" + item.select("a[rel='nofollow']")[0]['href']
    eara = item.select("a.address-eara")
    address_eara2 = item.select("span")
    for e in eara:
        print(e.text.strip())

    price = price_num + price_tag
    for eara in address_eara:
        address_info += eara.text.strip()
    for imgs in img_info:
        img_path = imgs['src']
        title = imgs['title']
    for size in size_span:
        size_info += size.text.strip()
    for eara in address_eara2:
        if "class" in str(eara) and '=""' in str(eara):
            title += eara.text.strip()
        else:
            pass

    print(price)
    print(address_info)
    print(size_info)
    print(title)
    print(time_info)
    print(img_path)
    print(href_info)
    db.insert_one("ganjicol", {"price_num": price_num, "price_tag": price_tag,"address_info": address_info, "size_info": size_info, "title": title, "time_info": time_info, "img_path": img_path, "herf_info": href_info})

db.close_conn()
#http://sz.ganji.com/fang/fang1/detail@puid=38080670253458" href="//sz.ganji.com/zufang/38080670253458x.shtml