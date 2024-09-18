import requests
from bs4 import BeautifulSoup

url = 'https://www.wta.org/go-hiking/trip-reports'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

#Example: Extracting hike titles
hike_titles = soup.find_all('h3', class_='documentFirstHeading') for title in hike_titles: print(title.get_text())