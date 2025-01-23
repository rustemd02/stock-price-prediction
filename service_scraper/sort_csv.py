import csv
from datetime import datetime

def parse_date(date_str):
    return datetime.strptime(date_str, '%Y-%m-%d')

input_file = '../static/news_sentiment.csv'
output_file = '../static/sorted_news_sentiment.csv'

with open(input_file, mode='r', newline='') as infile:
    reader = csv.DictReader(infile)
    data = list(reader)

sorted_data = sorted(data, key=lambda x: parse_date(x['TRADEDATE']))

with open(output_file, mode='w', newline='') as outfile:
    fieldnames = ['TRADEDATE', 'SentScore']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sorted_data)
