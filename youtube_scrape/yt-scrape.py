from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd
import argparse
import urllib.parse

# Set up argument parser to allow for passing search query
parser = argparse.ArgumentParser()
parser.add_argument("query", help="search query")
parser.add_argument("--source", default="CSPAN", help="source (default: CSPAN)")
args = parser.parse_args()

raw_query = args.query
source = args.source
escaped_query = urllib.parse.quote(f'"{raw_query}"')
url = f"https://www.youtube.com/@{source}/search?query={escaped_query}"

driver = webdriver.Chrome()
driver.get(url)

# Wait for the page to load
time.sleep(5)

# Scroll down the page to load additional results
scroll_pause_time = 2
scroll_height = driver.execute_script("return document.documentElement.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(scroll_pause_time)
    new_scroll_height = driver.execute_script("return document.documentElement.scrollHeight")
    if new_scroll_height == scroll_height:
        break
    scroll_height = new_scroll_height

# Extract links from the page
links = []
for link in driver.find_elements(By.XPATH, "//a[@id='video-title']"):
    href = link.get_attribute('href')
    title = link.get_attribute('title')
    links.append({'title': title, 'url': href})

# Create a DataFrame
df = pd.DataFrame(links)
print(f"{len(df)} links found")

# Print the DataFrame to a CSV file named after the search query
df.to_csv(f"youtube-results/{raw_query}.csv", index=False)
print(f"Results saved to {raw_query}.csv")

# Close the driver
driver.quit()
