import requests
from bs4 import BeautifulSoup
import pandas as pd

class data_acquisition_last_statements:

    #initialising class instance and calling the loading function
    def __init__(self):
        #calling the scraping function
        self.data_scraping()

    def data_scraping(self):

        # URL of the Texas Department of Criminal Justice death row info page
        url = 'https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html'

        #sending HTTP request to retrieving the webpage content by
        response = requests.get(url, verify=False) 

        #creating a beautiful soup object instance to parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        #find the main table that lists the executed offenders, identified by the HTML class
        offenders_table = soup.find('table', attrs={'class':'tdcj_table indent'})

        #extract all table rows (<tr> tags) and column headers from the offenders_table
        rows = offenders_table.find_all('tr') 
        headers = [header.text for header in rows[0].find_all('th')]

        data_rows = []  # list to hold all the row data

        # Iterate over all rows in the table but the header row
        for index, row in enumerate(rows):
            if index == 0: # Skip the header row
                continue  

            # Extract text from each cell (<td> tags) in the row
            cells = [cell.text for cell in row.find_all('td')]

            # Check for a link to the last statement in the current row
            last_statement_link = row.find('a', string='Last Statement')
            if last_statement_link:

                # create the full URL to fetch the last statement
                link_suffix = last_statement_link.get('href')
                if 'death_row' in link_suffix:
                    last_statement_url = 'https://www.tdcj.texas.gov/' + link_suffix
                else:
                    last_statement_url = 'https://www.tdcj.texas.gov/death_row/' + link_suffix
                
                # fetuch the last statement page with HTTP request

                last_statement_response = requests.get(last_statement_url, verify=False)
                last_statement_soup = BeautifulSoup(last_statement_response.text, 'html.parser')
                #find paragraphs of last statement
                paragraphs = last_statement_soup.find_all('p')
                for i, paragraph in enumerate(paragraphs):
                    if 'Last Statement:' in paragraph.text and i < len(paragraphs) - 1:
                        cells[2] = paragraphs[i + 1].text  # Replace the link in the cell with the last statement text
            
            # Append modified row (with the last statement text) to the data_rows list
            data_rows.append(cells)

        # Convert to pandas DataFrame with the extracted headers
        offender_df = pd.DataFrame(data_rows, columns=headers)

        # Save to a CSV file
        offender_df.to_csv('death_row_information.csv', index=False, encoding='utf-8')

        print("Data scraped and saved to 'death_row_information.csv'")
