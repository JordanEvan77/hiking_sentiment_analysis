import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By

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


driver = webdriver.Chrome()


url = 'https://www.wta.org/go-outside/trip-reports'
driver.get(url)
driver.implicitly_wait(10)
#-----------
page_source = driver.page_source
from bs4 import BeautifulSoup

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(page_source, 'html.parser')

# Extract hike titles and dates
hike_entries = soup.find_all('h3', class_='listitem-title')

hike_list = [entry.text.strip() for entry in hike_entries]
for hike in hike_list:
    print(hike)

#------
all_elements = driver.find_elements(By.CSS_SELECTOR, '*')


for element in all_elements[:1]:
    print('tag', element.tag_name, 'CLass', element.get_attribute('class'), 'TEXT', element.text)



hike_titles = driver.find_elements(By.CSS_SELECTOR, 'div.listitem-title a')


for title in hike_titles:
    print(title.text)

driver.quit()
