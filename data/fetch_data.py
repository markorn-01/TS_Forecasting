import nasdaqdatalink
import schedule
import time

def fetch_data():
    nasdaqdatalink.read_key(filename="/data/.corporatenasdaqdatalinkapikey")
    mydata = nasdaqdatalink.get("LBMA/GOLD")
    mydata.to_csv('data/gold/LBMA-GOLD.csv')

schedule.every().day.at("00:00").do(fetch_data)

while True:
    schedule.run_pending()
    time.sleep(1)