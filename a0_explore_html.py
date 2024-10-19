import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

import requests
from bs4 import BeautifulSoup

url = 'https://www.wta.org/go-outside/trip-reports'
url2 = 'https://www.wta.org/go-hiking/trip-reports/trip_report-2024-10-19.114407414301'
response = requests.get(url)

# This is a main trip report page, and after clicking on each report (trail name and date) the
# details are underneath

# I am interested in:
    # Trail Report By
    # Type of Hike
    # Trail Conditions
    # Road
    # Bugs
    # Snow
    # and the actual written report.
    # I DONT care about the comments, or bottom of the page links.

    # BONUS: IF I can get to the Trails Hiked page, I would also like:
        # Length
        # Elevation Gain
        # Highest Point
        # Maybe grab other category characteristics? later?

page = requests.get(url)
print(page.text)

soup = BeautifulSoup(page.content, "html.parser")
results = soup.find(id="trip-reports")
print(results.prettify())
#----
hike_titles = soup.find_all('h3', class_='listitem-title')


if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    trip_reports = soup.find_all('h2', class_='trip-report-title')

    for report in trip_reports:
        print('report', report.text)
else:
    print('Failed to retrieve the webpage')





