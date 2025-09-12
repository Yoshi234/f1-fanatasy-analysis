# import dependencies
# from playwright.sync_api import sync_playwright
# from bs4 import BeautifulSoup 
import pandas as pd
import re
import numpy as np
import json


def copilot_price_data_output():
    '''
    Use this prompt to fill in drivers and constructors dictionaries that can 
    be used to source pricing data. This saves a little bit of time, but trying 
    to "save" further time with API sourcing will be much more time consuming and 
    complicated. 

    ```prompt
    TASK: Read the tables on this page and output them as python dictionaries which can be read as pandas dataframes. Please ensure that Tier A and Tier B assets for both drivers and constructors are listed as separate dataframes. As a rule of thumb - any group of data that gets its own table should get its own column. 
    
    EXAMPLE: The Tier A drivers dictionary should be formatted as
    driversA = { "driver": ["PIA", "RUS", "LEC", "NOR", "HAM", "VER"], "price/$": [26.4, 22.4, 23.9, 31.4, 23.8, 28.2], "R13 Pts": [42, 19, 30, 39, 40, 30], "R14 Pts": [29, 37, 20, 36, 4, 11], "-0.3": [-24, -16, -7, -19, -1, 10], "-0.1": [-23, -15, -6, -18, 0, 11], "+0.1": [1, 5, 15, 10, 21, 36], "+0.3": [24, 25, 36, 38, 42, 61] } 
    
    CONTEXT: In this dictionary, 'driver' corresponds to the "DR" column, and price corresponds to the "$" column under "Tier A". Instead of labeling every column as "Pts" for the remaining columns, the top value "R13", "R14", etc. is used instead.
    ```

    Use these as a starter template and fix things as necessary. 
    '''
    driversA = {
        "driver": ["PIA", "RUS", "VER", "LEC", "NOR", "HAM"],
        "price/$": [26.7, 22.5, 28.1, 23.6, 31.3, 23.5],
        "R14 Pts": [29, 37, 11, 20, 36, 4],
        "R15 Pts": [45, 19, 29, -13, -5, -16],
        "-0.3": [-26, -16, 10, 35, 25, 54],
        "-0.1": [-25, -15, 11, 36, 26, 55],
        "+0.1": [-1, 5, 36, 57, 54, 76],
        "+0.3": [22, 25, 61, 78, 82, 97]
    }
    driversB = {
        "driver": ["BOR", "LAW", "ALO", "SAI", "STR", "HUL", "ALB", "OCO", "HAD", "COL", "ANT", "GAS", "BEA", "TSU"],
        "price/$": [5.1,   5.1,   8.1,   5.7,   10.3,  9.8,   11.8,  8.7,   5.7,   4.5,   15.7,  4.8,   7.3,   10.6],
        "R13 Pts": [7,     9,     7,     6,     9,     7,     14,    3,     -7,    -2,    21,    -14,   7,     0],
        "R14 Pts": [24,    10,    17,    6,     13,    14,    12,    6,      4,    -4,    10,     0,   -18,    3],  
        "-0.6": [  -15,   -1,    -16,    3,    -16,    -5,    -18,  -6,     -27,  np.nan, 17,   np.nan, 4,    5],
        "-0.3": [ np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, np.nan, np.nan, 8,   np.nan,np.nan],
        "-0.2": [  -14,    0,    -15,    4,    -15,    -4,    -17,  -5,     -26,  np.nan, 18,      9,    5,    6],
        "0.0": [  np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, np.nan,  5, np.nan, np.nan,np.nan,np.nan],
        "+0.2": [  -10,    4,     -8,    9,    -6,      5,     -7,    3,    -21,    6,    32,     13,   11,    15],
        "+0.6": [  -5,     9,     0,    14,    3,       14,     4,    11,   -16,    10,   46,     18,   18,    24]
    }
    constructorsA = {
        "constructor": ["MCL", "MER", "RED", "FER"],
        "price/$": [34.5, 26.0, 28.7, 30.4],
        "R14 Pts": [100, 54, 39, 39],
        "R15 Pts": [52, 30, 47, -4],
        "-0.3": [-90, -37, -34, 20],
        "-0.1": [-89, -36, -33, 21],
        "+0.1": [-58, -13, -8, 48],
        "+0.3": [-28, 10, 17, 74]
    }
    constructorsB = {
        "constructor": ["AST", "VRB", "KCK", "WIL", "HAA", "ALP"],
        "price/$": [11.9, 11.8, 9.9, 16.1, 12.4, 7.7],
        "R14 Pts": [45, 29, 35, 24, -9, 2],
        "R15 Pts": [41, 38, 14, 35, 43, 14],
        "-0.6": [-65, -46, -32, -30, -12, -3],
        "-0.2": [-64, -45, -31, -29, -11, -2],
        "+0.2": [-53, -35, -22, -15, 0, 5],
        "+0.6": [-43, -24, -13, -1, 11, 12]
    }
    
    return (
        pd.DataFrame(driversA), pd.DataFrame(driversB), 
        pd.DataFrame(constructorsA), pd.DataFrame(constructorsB)
    )


def parse_raw_output(raw_text):
    string_replacements = [
        ("__next_f.push", ""),
        ("\\n", ""), ("\n", ""), 
        ("null", '"null"'), ("\\", "")
    ]
    strip_tokens = ["(", ")"]

    for s_pair in string_replacements:
        raw_text = raw_text.replace(s_pair[0], s_pair[1])

    for s_token in strip_tokens:
        raw_text = raw_text.strip(s_token)


def extract_tables_from_html(
    html_text,
    tier_label
):
    '''
    Consumes html output generated with playwright and 
    outputs a series of dataframes corresponding to each of the 
    tables at a given webpage

    Args:
    - html_text (str): HTML encoded in string format for parsing by 
      beautiful soup
    Returns:
    - dfs (list[pd.DataFrame]): A list of pandas dataframes each 
      corresponding to the tables stored at the webpage from which
      the HTML content was scraped
    '''
    pattern = rf"{tier_label}.*?DF \ $ Pts.*?(?=Tier|Find a constructor|\Z)"
    match = re.search(pattern, html_text, re.DOTALL)
    if not match: return None
    block = match.group(0)
    # soup = BeautifulSoup(html, "html.parser")
    # tables = soup.findall("table") # find all table objects
    # dfs = []

    # for table in tables:
    #     rows = table.find_all("tr")
    #     table_data = []
    #     for row in rows:
    #         cols = row.find_all("td")
    #         if cols:
    #             table_data.append([col.get_text(strip=True) for col in cols])

    #     df = pd.DataFrame(table_data)
    #     dfs.append(df)
    
    # return dfs


def scrape_tables(
    webpage:str = "https://f1fantasytools.com/budget-builder",
    headless:bool = True,
    dest_path:str = "../results/zandfort",
    dynamic_wait_str:str = "Tier A (>=18.5 M)"
):
    '''
    Scrapes tables from a given webpage and uses beautiful soup to 
    parse the webpage for tables which might be either dynamically 
    generated or static

    Args:
    - webpage (str): The webpage link which you would enter into a 
      browser
    - headless (bool): If true, does open the browser for explicit 
      viewing, but instead runs the process in the background. Typically
      this is more efficient
    - dest_path (path[str]): The path to save the output tables to
    - dynamic_wait_str (str): A string to wait for viewing in the output
      before loading all of the HTML content - this prevents the loading
      of non-hydrated HTML outputs
    Returns:
    - None
    '''
    print("[INFO]: Starting Playwright")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=headless)
        page = browser.new_page()
        page.goto(webpage, timeout=60000)
        page.wait_for_timeout(1000) # wait until tables are generated

        raw_text = page.content()
        # print(raw_text)
        print("=========================")
        matches = re.findall(r"__next_f\.push\(\[.*?\]\)", raw_text)

        browser.close()
        
    print("[INFO]: Close Browser")

    # print(raw_text)
    # driver_lines = [line for line in raw_text.splitlines() if re.match(r"^[A-Z]{3} \d", line)]
    # print(driver_lines)
    print(matches[-1])
    print(json.loads(matches[-1]))
    exit()

    dfs = extract_tables_from_html(html, dynamic_wait_str)
    
    for df in dfs: 
        print(df)


if __name__ == "__main__":
    for x in copilot_price_data_output():
        print(x)