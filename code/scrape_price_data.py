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
        "driver": ["VER", "RUS", "NOR", "HAM", "LEC", "PIA"],
        "price/$": [28.5,  22.7,  30.9,  22.9,  23.0,  26.9],
        "R16 Pts": [46, 19, 37, 26, 21, 25],
        "R17 Pts": [45, 30, 12, 12, 6, -16],
        "-0.3": [-40, -8, 7, 3, 14, 39],
        "-0.1": [-39, -7, 8, 4, 15, 40],
        "+0.1": [-14, 13, 35,24,36, 64],
        "+0.3": [12,  33, 62,45,56, 88],
    }
    driversB = {
        "driver": ["SAI", "LAW", "BOR", "GAS", "HAD", "COL", "ALB", "ANT", "TSU", "BEA", "ALO", "OCO", "STR", "HUL"],
        "price/$": [5.7,   5.9,   5.9,   4.5,   6.9,   4.5,   13.0,  14.9,  9.8,   7.3,   7.3,   8.3,   9.9,   8.6],
        "R16 Pts": [3,     6,     13,    5,     12,    0,     15,    6,     0,     3,     -15,   1,     -1,   -20],
        "R17 Pts": [33,    17,    7,     5,     4,     1,     13,    22,    15,    6,     -3,    6,     -2,    4],
        "-0.6": [-26,     -13,    -10, np.nan,  -4,  np.nan,  -5,    -1,    2,     4,     31,    8,     20,   31],
        "-0.2": [-25,     -12,    -9,  np.nan,  -3,  np.nan,  -4,    0,     3,     5,     32,    9,     21,   32],
        "0.0": [np.nan,  np.nan, np.nan, 2,   np.nan,  11,   np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
        "+0.2": [-20,      -7,     -4,   3,     3,     12,     8,    13,    12,    11,    38,    16,    30,   40],
        "+0.6": [-15,      -1,     -2,   7,     9,     16,     19,   26,    21,    18,    45,    23,    39,   47]
    }
    constructorsA = {
        "constructor": ["RED", "MCL", "MER", "FER"],
        "price/$": [29.3, 35.1, 26.6, 30.6],
        "R16 Pts": [56, 97, 37, 67],
        "R17 Pts": [85, 6, 67, 28],
        "-0.3": [-88, -40, -56, -40],
        "-0.1": [-87, -49, -55, -39],
        "+0.1": [-61, -8,  -32, -12],
        "+0.3": [-36, 23,  -8,  15],
    }
    constructorsB = {
        "constructor": ["VRB", "WIL", "ALP", "KCK", "HAA", "AST"],
        "price/$": [13.0, 17.3, 8.1, 10.7, 13.2, 12.3],
        "R16 Pts": [27, 23, 6, -2, 7, -11],
        "R17 Pts": [41, 46, 7, 17, 13, -2],
        "-0.6": [-45, -38, 1, 4, 3, 35],
        "-0.2": [-44, -37, 2, 5, 4, 36],
        "+0.2": [-32, -22, 9, 14, 16, 47],
        "+0.6": [-21, -7, 17, 24, 28, 58],
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