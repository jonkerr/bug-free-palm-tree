'''
Based on Naomi's work in S&P500_data_v1.ipynb
'''
import requests
from bs4 import BeautifulSoup
import pandas as pd

class MultplDataScraper():
    def __init__(self) -> None:
        self.entries = [
            (
                "https://www.multpl.com/s-p-500-historical-prices/table/by-month",
                "S&P500 Price"
            ),
            (
                "https://www.multpl.com/inflation-adjusted-s-p-500/table/by-month",
                "S&P500 Price - Inflation Adjusted"
            ),
            (
                "https://www.multpl.com/s-p-500-dividend-yield/table/by-month",
                "S&P500 Dividend Yield"
            ),
            (
                "https://www.multpl.com/s-p-500-pe-ratio/table/by-month",
                "S&P500 PE ratio"
            ),
            (
                "https://www.multpl.com/s-p-500-earnings/table/by-year",
                "S&P500 Earnings"
            ),
            (
                "https://www.multpl.com/s-p-500-earnings-yield/table/by-month",
                "S&P500 Earnings Yield"
            ),
            (
                "https://www.multpl.com/shiller-pe/table/by-month",
                "Shiller PE Ratio"
            ),
            (
                "https://www.multpl.com/case-shiller-home-price-index-inflation-adjusted/table/by-month",
                "case Shiller Home Price Index"
            ),
            (
                "https://www.multpl.com/inflation/table/by-month",
                "Inflation Rate"
            ),
        ]


    def scrape_and_process_data(self, url, column_name):
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the HTML content of the page
            soup = BeautifulSoup(response.content, "html.parser")

            # Find the table containing the data
            table = soup.find("table")

            # Extract table headers
            headers = [th.text.strip() for th in table.find_all("th")]

            # Extract table rows
            rows = []
            for row in table.find_all("tr")[1:]:
                rows.append([td.text.strip() for td in row.find_all("td")])

            # Create a Pandas DataFrame
            df = pd.DataFrame(rows, columns=headers)

            # Convert the 'Date' column to datetime object
            df['Date'] = pd.to_datetime(df['Date'])

            # Convert the 'Value' column to float for numeric values, keeping strings as is
            def convert_to_float(value):
                try:
                    # Remove non-numeric characters and commas
                    cleaned_value = ''.join(
                        char for char in value if char.isdigit() or char == '.')
                    return float(cleaned_value)
                except ValueError:
                    return value

            df[column_name] = pd.to_numeric(
                df['Value'].apply(convert_to_float), errors='coerce')

            # Set 'Date' column as the index
            df.set_index('Date', inplace=True)

            # Drop the original 'Value' column if needed
            df.drop(columns=['Value'], inplace=True)

            return df

        else:
            print(
                f"Failed to retrieve the webpage. Status Code: {response.status_code}")
            return None

    def get_data(self):
        dfs = [self.scrape_and_process_data(entry[0], entry[1]) for entry in self.entries]
        return pd.concat(dfs, axis=1)
