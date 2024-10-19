import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

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

# parser soup
soup_outer = BeautifulSoup(page_source, 'html.parser')

# titles
hike_entries = soup_outer.find_all('h3', class_='listitem-title')

# now use them as links
raw_df = pd.DataFrame({'Hike Name': [],
                       'Trail Report By': [],
                       'Type of Hike': [],
                       'Trail Conditions': [],
                       'Road': [],
                       'Bugs': [],
                       'Snow': []
                       # and the actual written report
                       })
for entry in hike_entries:
    link = entry.find('a')
    if link:
        print(f'Navigating to: {link.text.strip()}')
        href = link.get('href')
        driver.get(href)

        # load time
        time.sleep(5)

        # get needed parts
        page_source = driver.page_source
        soup_middle = BeautifulSoup(page_source, 'html.parser')

        # Extract the specific details
        details = soup_middle.find_all('div', class_='trip-condition')
        detail_dict = {detail.find('h4').text.strip(): detail.find('span').text.strip()
            for detail in details}
        hike_title = soup_middle.find('span', id='breadcrumbs-current').text.strip() if \
            soup_middle.find('span', id='breadcrumbs-current') else 'N/A'
        trail_report_by = soup_middle.find('span', class_='wta-icon-headline__text').text.strip() \
            if soup_middle.find('span', class_='wta-icon-headline__text') else 'N/A'
        report_text = soup_middle.find('div', id='tripreport-body-text').get_text(
            strip=True) if soup_middle.find('div', id='tripreport-body-text') else 'N/A'

        # ONE LAYER FURTHER IN!
        trails_hiked_link = soup_middle.find('div', class_="related-hike-links")
        link = trails_hiked_link.find('a')
        if link:
            print(f'Navigating to inner details: {link.text.strip()}')
            href = link.get('href')
            driver.get(href)

            # load time
            time.sleep(5)

            # get needed parts
            page_source = driver.page_source
            soup_inner = BeautifulSoup(page_source, 'html.parser')


            stats = soup_inner.find_all('div', class_='hike-stats__stat')
            for stat in stats:
                detail_dict[stat.find('dt').text.strip()] = stat.find('dd').text.strip()

            region_info = soup_inner.find('div', class_='region')
            if region_info:
                trailhead_region = region_info.find('span',  class_='region').text.strip()
            sidebar = soup_inner.string('div', class_='wta-sidebar-layout__sidebar')
            rating_section = sidebar.find('div', id='hike-rating')
            if rating_section:
                rating = float(rating_section.find('div', class_='current-rating').text.strip(
                ).split()[0])

        #Now a temp df
        detail_df = pd.DataFrame({
            'Hike Name': [hike_title],
            'Trail Report By': [trail_report_by],
            'Type of Hike': [detail_dict['Type of Hike']],
            'Trail Conditions': [detail_dict['Trail Conditions']],
            'Road': [detail_dict['Road']],
            'Bugs': [detail_dict['Bugs']],
            'Snow': [detail_dict['Snow']],
            'Report Text':[report_text]
        })
        raw_df = pd.concat([raw_df, detail_df], axis=0)
        # now in one final layer
        
        #back out
        driver.back()
        driver.implicitly_wait(10)


driver.quit()


raw_df.to_csv('hiking_reports.csv', index=False)
print(raw_df.shape, 'saved!')