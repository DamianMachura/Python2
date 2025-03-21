import requests
import sqlite3

API_URL = 'https://api.lotto.pl/v1/lotteries/draw-results/by-gametype?gameType=MINI_LOTTO&size=100'
DB_NAME = 'mini_lotto.db'

def fetch_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f'Błąd podczas pobierania danych: {response.status_code}')

def create_table(conn):
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS mini_lotto (
                draw_date TEXT PRIMARY KEY,
                numbers TEXT
            )
        ''')

def save_to_db(conn, data):
    with conn:
        for draw in data:
            draw_date = draw['drawDate']
            numbers = ','.join(map(str, draw['results']))
            conn.execute('''
                INSERT OR REPLACE INTO mini_lotto (draw_date, numbers)
                VALUES (?, ?)
            ''', (draw_date, numbers))

def main():
    data = fetch_data(API_URL)
    
    conn = sqlite3.connect(DB_NAME)
    create_table(conn)
    save_to_db(conn, data)
    conn.close()
    print('Dane zostały zapisane w bazie danych.')

if __name__ == '__main__':
    main()
