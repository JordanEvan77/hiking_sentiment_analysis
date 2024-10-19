import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

import requests
from bs4 import BeautifulSoup

url = 'https://www.wta.org/go-hiking/trip-reports'
response = requests.get(url)


if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    trip_reports = soup.find_all('h2', class_='trip-report-title')

    for report in trip_reports:
        print('report', report.text)
else:
    print('Failed to retrieve the webpage')





