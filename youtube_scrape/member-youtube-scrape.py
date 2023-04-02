import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Load the member data from the CSV file
df = pd.read_csv('members.csv')

# Create a new Chrome driver
driver = webdriver.Chrome()

# Define the base YouTube search URL
search_url = 'https://www.youtube.com/results?search_query='

# Iterate over the rows in the DataFrame
primary_channels = []
secondary_channels = []
for index, row in df.iterrows():
    first_name = row['first_name']
    last_name = row['last_name'].split(" ")[0]
    search_query = f'Congress+{first_name}+{last_name}'

    # Construct the full search URL, including only channels
    url = search_url + search_query + "&sp=EgIQAg%253D%253D"

    # Load the search results page
    driver.get(url)

    # Wait for the page to fully load
    time.sleep(5)

    # Find the first search result
    try:
        channel_links = driver.find_elements(By.CSS_SELECTOR, 'a.channel-link')
        if len(channel_links) == 0:
            print(f'No results found for {first_name} {last_name}')
            primary_channels.append('')
            secondary_channels.append('')
        else:
            primary_channels.append(channel_links[0].get_attribute('href'))
            if len(channel_links) > 2:
                secondary_channels.append(channel_links[2].get_attribute('href'))
            else:
                secondary_channels.append('')
    except:
        print(f'Error finding channels for {first_name} {last_name}')
        primary_channels.append('')
        secondary_channels.append('')

# Close the Chrome driver
driver.quit()

df['primary_youtube'] = primary_channels
df['secondary_youtube'] = secondary_channels
df.to_csv('youtube.csv', header=True, index=False)

