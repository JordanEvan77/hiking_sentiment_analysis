# hiking_sentiment_analysis
Using data from WTA trip reports, review the reports, do sentiment analysis, and pair with hiking characteristics. Then recommend future hikes per user


Reports page: https://www.wta.org/go-outside/trip-reports

import requests
from bs4 import BeautifulSoup

url = 'https://www.wta.org/go-hiking/trip-reports'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Example: Extracting hike titles
hike_titles = soup.find_all('h3', class_='documentFirstHeading')
for title in hike_titles:
    print(title.get_text())


Guide: https://realpython.com/beautiful-soup-web-scraper-python/
Examples: https://brightdata.com/blog/how-tos/beautiful-soup-web-scraping

Valuable data; 
Type of Hike:Day hike
Trail Conditions:Trail in good condition
Road:Road suitable for all vehicles
Bugs:No bugs
Snow:Snow free
Region: Issaquah Alps
Sentiment: Did they like the trail?
Reason: Why did they like the trail?
Bonus:
IF I can get the hike details (length, elevation gain and highest point), that would help characterize the sentiment further.
First goal: Predict if they will like it.
Second goal: recommend what they might like the most.
