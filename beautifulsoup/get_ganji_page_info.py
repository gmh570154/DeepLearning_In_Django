import requests
from bs4 import BeautifulSoup

headers = {
    'user-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.94 Safari/537.36'
}
home_url = 'http://sz.ganji.com/xixiangsz/zufang/0/'
wb_data = requests.get(home_url,headers = headers)
soup = BeautifulSoup(wb_data.text,'html.parser')
utag = soup.select('div.f-list-item.ershoufang-list')

for item in utag:
    price = ""
    address_info = ""
    size_info = ""
    title = ""
    img_path = ""
    #print(item.getText)
    price_num = item.select("span.num")[0].text.strip()
    price_tag = item.select("span.yue")[0].text.strip()
    address_eara = item.select("span.address-eara")
    img_info = item.select("img")
    size_span = item.select("dd.size > span")

    price = price_num + price_tag
    for eara in address_eara:
        address_info += eara.text.strip()
    for imgs in img_info:
        img_path = imgs['src']
        title = imgs['title']
    for size in size_span:
        size_info += size.text.strip()

    print(price)
    print(address_info)
    print(size_info)
    print(title)
    print(img_path)
