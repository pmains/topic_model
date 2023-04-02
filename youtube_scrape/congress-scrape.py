import pandas as pd
from selenium import webdriver
import time
from bs4 import BeautifulSoup

driver = webdriver.Chrome()

df = pd.DataFrame()

for page in range(1,8):
    url = f'https://www.congress.gov/members?q=%7B%22congress%22%3A%5B118%2C117%5D%7D&page={page}'
    driver.get(url)
    
    # Sleep for 5 seconds to allow the page to fully load
    time.sleep(5)
    
    # Parse the HTML content of the page
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    members = []
    for member in soup.find_all('div', class_='quick-search-member'):
        try:
            name = member.find('div', class_='member-image').find('img')['alt']
        except:
            name = 'Unknown, Unknown'
    
        state = member.find('strong', text='State:').find_next_sibling('span').text.strip()
        try:
            district = member.find('strong', text='District:').find_next_sibling('span').text.strip()
        except:
            print(f"Cannot find district for {name}")
            district = None
        party = member.find('strong', text='Party:').find_next_sibling('span').text.strip()
        party = party.split()[0].upper()
    
        # Extract the first and last name from the full name
        name_list = name.split(', ')
        first_name, last_name = name_list[:2]
        if len(name_list) > 2:
            print(f'{name} has more than 2 tokens')
        members.append({
            'first_name': first_name,
            'last_name': last_name,
            'state': state,
            'party': party,
            'district': district
        })

    df = pd.concat([df, pd.DataFrame(members)])

df = df.drop_duplicates()
print(f'{len(df)} Members')
df.to_csv('house.csv', header=True, index=False)

# Close the Selenium driver
driver.quit()
