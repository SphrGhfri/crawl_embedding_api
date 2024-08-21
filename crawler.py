from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import html2text
from typing import Union, Literal

# Set up Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Enable headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument(
    "--window-size=1920,1080"
)  # Set window size for headless mode
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
)
chrome_options.add_argument("--enable-javascript")

# Initialize the Chrome WebDriver
driver = webdriver.Chrome(options=chrome_options)

# Function to clean the HTML content by removing links and code blocks
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove all links
    for a in soup.find_all("a"):
        a.unwrap()  # Remove the <a> tag but keep the content inside
    
    # Remove all code blocks
    for code in soup.find_all("code"):
        code.decompose()  # Completely remove the <code> tag and its content
    
    return str(soup)

# Function to extract and save table data
def extract_table_and_save(element: Union[Literal['table_with_header'], Literal['table_without_header']] = 'table_without_header'):
    # Wait for the specific element to be present on the final page
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.TAG_NAME, "footer"))
    )

    # Once the element is found, get the page source
    page_source = driver.page_source

    # Parse the final page content with BeautifulSoup
    soup = BeautifulSoup(page_source, "html.parser")
    
    if element == 'table_with_header':
        main_table = soup.select_one(
            "#frmSearch > div:nth-child(2) > div > table.border-list.table.table-striped.table-hover"
        )  # Selects the first table
    else:
        main_table = soup.select_one(
            "#frmSearch > div:nth-child(2) > div > table.border-list.table.table-striped.table-hover > tbody:nth-child(2)"
        )  # Selects the first table

    # Clean the HTML to remove links and code blocks
    cleaned_html = clean_html(str(main_table))

    # Convert the cleaned HTML table to Markdown
    table_markdown = html2text.html2text(cleaned_html)

    # Save the Markdown content to a file
    with open("crawled_table.md", "a", encoding="utf-8") as file:
        file.write(table_markdown[:-1])

try:
    # Loop through pages 1 to 5
    for page in range(1, 6):
        # Construct the URL for the specific page
        url = f"https://qavanin.ir/?PageNumber={page}"
        # Navigate to the initial URL
        driver.get(url)

        # Wait a bit to ensure the page is fully loaded
        driver.implicitly_wait(15)

        # Extract and save the table data
        if page == 1:
            extract_table_and_save("table_with_header")
        else:
            extract_table_and_save()

finally:
    # Clean up and close the WebDriver
    driver.quit()
