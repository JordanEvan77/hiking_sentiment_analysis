import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
from bs4 import BeautifulSoup

data_dir = r'C:\Users\jorda\OneDrive\Desktop\PyCharm Community Edition 2021.2.2\EXTERNAL DATA ' \
           r'SCIENCE PROJECTS 2023\Hiking Sentiment\data\\'

driver = webdriver.Chrome()
url = 'https://www.wta.org/go-outside/trip-reports'
driver.get(url)
driver.implicitly_wait(5)

raw_df = pd.DataFrame(
    {'Hike Name': [], 'Trail Report By': [], 'Type of Hike': [], 'Trail Conditions': [], 'Road': [],
     'Bugs': [], 'Snow': [], 'Report Text': [], 'Region': [], 'Elevation': [], 'Highest Point': [],
     'Difficulty': [], 'Rating': [], 'Key Features': [], 'Date': []})

i = 0
start_time = time.time()
dead_links = 0
while True:
    page_source = driver.page_source
    soup_outer = BeautifulSoup(page_source, 'html.parser')
    hike_entries = soup_outer.find_all('h3', class_='listitem-title')  # find next 100

    for entry in hike_entries:
        i += 1
        print('Count of hikes', i, '/', len(hike_entries))
        link = entry.find('a')
        if link:
            print(f'Navigating to: {link.text.strip()}')
            date = link.text.strip().split('—')[1]  # get date
            href = link.get('href')
            try:
                driver.get(href)
            except:
                print('Bad get, skip')
                continue
            # load time
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, 'tripreport-body-text')))
            except TimeoutException:
                dead_links += 1
                print(f'*** DEAD LINK MOVE ON, COUNT: {dead_links}')
                continue
            # get needed parts
            page_source = driver.page_source
            soup_middle = BeautifulSoup(page_source, 'html.parser')
            # Extract the specific details
            details = soup_middle.find_all('div', class_='trip-condition')
            detail_dict = {detail.find('h4').text.strip(): detail.find('span').text.strip() for
                           detail in details}
            hike_title = soup_middle.find('span',
                                          id='breadcrumbs-current').text.strip() if soup_middle.find(
                'span', id='breadcrumbs-current') else 'N/A'
            trail_report_by = soup_middle.find('span',
                                               class_='wta-icon-headline__text').text.strip() if soup_middle.find(
                'span', class_='wta-icon-headline__text') else 'N/A'
            report_text = soup_middle.find('div', id='tripreport-body-text').get_text(
                strip=True) if soup_middle.find('div', id='tripreport-body-text') else 'N/A'
            # ONE LAYER FURTHER IN!
            trails_hiked_link = soup_middle.find('div', class_="related-hike-links")
            link = trails_hiked_link.find('a')
            if link == None:
                continue  # empty link, next dataset?
            if link:
                href = link.get('href')
                driver.get(href)
                try:
                    WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.hike-stats__stat')))
                except TimeoutException:
                    dead_links +=1
                    print(f'*** DEAD LINK MOVE ON, COUNT: {dead_links}')
                    continue
                # get needed parts
                page_source = driver.page_source
                soup_inner = BeautifulSoup(page_source, 'html.parser')
                stats = soup_inner.find_all('div', class_='hike-stats__stat')
                detail_dict['Elevation Gain'] = 'N/A'
                detail_dict['Highest Point'] = 'N/A'
                detail_dict[
                    'Calculated Difficulty\n                            \n\nAbout Calculated Difficulty'] = 'N/A'
                for stat in stats:
                    detail_dict[stat.find('dt').text.strip()] = stat.find('dd').text.strip()
                region_info = soup_inner.find('div', class_='region')
                rating = 'N/A'
                trailhead_region = 'N/A'
                if region_info:
                    trailhead_region = region_info.find('span', class_='region').text.strip()
                rating_div = soup_inner.find('div', class_='current-rating')
                rating = rating_div.text.strip()
                # now for the icon list:
                icons_list = []
                icons_section = soup_inner.find('ul', class_='wta-icon-list')
                if icons_section:
                    icons = icons_section.find_all('li')
                    for icon in icons:
                        icon_label = icon.find('span', class_='wta-icon__label').text.strip()
                        icons_list.append(icon_label)
            # Now a temp df
            detail_df = pd.DataFrame({
                'Hike Name': [hike_title], 'Trail Report By': [trail_report_by],
                'Type of Hike': [detail_dict['Type of Hike']],
                'Trail Conditions': [detail_dict['Trail Conditions']],
                'Road': [detail_dict['Road']], 'Bugs': [detail_dict['Bugs']],
                'Snow': [detail_dict['Snow']], 'Report Text': [report_text],
                'Region': [trailhead_region], 'Elevation': [detail_dict['Elevation Gain']],
                'Highest Point': [detail_dict['Highest Point']], 'Difficulty': [detail_dict[
                                                                                    'Calculated Difficulty\n                            \n\nAbout Calculated Difficulty']],
                'Rating': [rating], 'Key Features': [icons_list], 'Date': date
            })
            raw_df = pd.concat([raw_df, detail_df], axis=0)
            # now in one final layer
            # back out
            driver.back()
            driver.implicitly_wait(5)

    # Check if there is a "Next 100 items" button
    next_link = soup_outer.find('li', class_='next').find('a')[
        'href']  # <span class="label">Next 100 items</span>
    if next_link:
        driver.get(next_link)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.listitem-title')))
        print(
            f'Next loop begins with {i + 1} and time spent {(time.time() - start_time) / 60} minutes')
    else:
        print('clicked through all')
        break
    if i % 5000 == 0:
        raw_df.to_csv(
            data_dir + 'raw\hiking_reports.csv',
            index=False)

raw_df.to_csv(
    data_dir + 'raw\hiking_reports.csv',
    index=False)
driver.quit()
print(raw_df.shape, 'saved!')
