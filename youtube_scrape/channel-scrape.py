import time
import sys

from multiprocessing import Pool
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def find_all_video_links(channel_url):
    print("Scraping channel: " + channel_url)

    videos_url = f"{channel_url}/videos"
    try:
        driver = webdriver.Chrome()
        driver.get(videos_url)
        wait = WebDriverWait(driver, 10)

        # Scroll down to lazy load videos
        scroll_pause_time = 1
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(scroll_pause_time)
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Find all video links
        video_links = []
        video_elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'a#thumbnail.ytd-thumbnail')))
        for video_element in video_elements:
            # link_element = video_element.find_element(By.CSS_SELECTOR, 'a#thumbnail')
            link_url = video_element.get_attribute('href')
            if link_url is not None:
                video_links.append((channel_url,link_url))
    except:
        driver.quit()
        return {'channel_url': channel_url, 'status': 'error'}
    finally:
        driver.quit()

    return {'channel_url': channel_url, 'status': 'success', 'video_links': video_links}


def main(file_name=None):
    if file_name is None:
        file_name = "youtube.xlsx"

    df = pd.read_excel(file_name)
    # Get all official and campaign channels
    channel_urls = set(df.official_channel.dropna()).union(set(df.campaign_channel.dropna()))

    with Pool(processes=15) as pool:
        results = pool.map(find_all_video_links, channel_urls)

    video_file_name = "videos.csv"
    try:
        df_video_links = pd.read_csv(video_file_name)
    except FileNotFoundError:
        df_video_links = pd.DataFrame(columns=['channel_url', 'video_url'])

    df_errors = pd.DataFrame([], columns=['channel_url', 'status'])

    # Consolidate results into a list of tuples
    for result in results:
        if result['status'] == 'success':
            channel_links = result['video_links']
            for channel_link in channel_links:
                df_video_links = pd.concat([
                    df_video_links, pd.DataFrame([channel_link], columns=['channel_url', 'video_url'])
                ])
        elif result['status'] == 'error':
            print(f"Error scraping channel: {result['channel_url']}")
            df_errors = pd.concat([
                df_errors, pd.DataFrame([result], columns=['channel_url', 'status'])
            ])

    # Save results to CSV
    df_video_links.to_csv(video_file_name, index=False)
    df_errors.to_csv('errors.csv', index=False)


if __name__ == '__main__':
    # Read the file name from the command line
    if len(sys.argv) > 1:
        file_name_arg = sys.argv[1]
    else:
        file_name_arg = None

    main(file_name_arg)
