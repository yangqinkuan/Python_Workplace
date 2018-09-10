import urllib.request
url = "http://www.douban.com/"
webPage = urllib.request.urlopen(url)
data = webPage.read()
data = data.decode('UTF-8')
print(data)