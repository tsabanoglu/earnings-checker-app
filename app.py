import sqlite3
import pandas as pd
import logging
import yfinance as yf
import json
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import streamlit as st

# Regular expression pattern to match valid stock tickers
ticker_pattern = re.compile(r'^[A-Z.]{1,7}$')

# Function to scrape tickers of consumer companies
def scrape_tickers(url, category):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    ticker_elements = soup.find_all('a', href=True)

    tickers = []
    for element in ticker_elements:
        href = element['href']
        text = element.text.strip()
        if href.startswith('/stocks/') and href.count('/') == 3:
            if ticker_pattern.match(text):
                tickers.append({
                    'ticker': text.replace('.', '-'),  # Replace dots with dashes
                    'category': category
                })
    return tickers

def save_tickers_to_json(filename):
    urls = [
        ('https://stockanalysis.com/stocks/sector/consumer-staples/', 'Consumer Staples'),
        ('https://stockanalysis.com/stocks/sector/consumer-discretionary/', 'Consumer Discretionary')
        # Add more URLs as needed
    ]
    all_tickers = []
    for url, category in urls:
        all_tickers.extend(scrape_tickers(url, category))

    with open(filename, 'w') as f:
        json.dump(all_tickers, f)

# Save tickers to JSON file
save_tickers_to_json('tickers.json')

# --- Database Initialization ---

def initialize_database(db_path):
    """Create the initial database schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS quarterly_revenue (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        name TEXT,
        quarter TEXT,
        revenue REAL,
        earnings_date DATE,
        revenue_growth REAL,
        revenue_average REAL,
        gross_profit REAL,
        net_income REAL,
        ebitda REAL,
        employee_count INTEGER,
        trailing_pe REAL,
        category TEXT,
        UNIQUE(ticker, quarter)  -- Ensure no duplicate records for the same ticker and quarter
    )
    ''')

    conn.commit()
    conn.close()

def load_tickers_from_db(db_path):
    """Load the list of tickers from the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT ticker FROM quarterly_revenue')
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers

# --- Insert Data Functions ---

def insert_quarterly_revenue(conn, ticker, name, quarter, revenue, earnings_date, revenue_growth, revenue_average, gross_profit, net_income, ebitda, employee_count, trailing_pe, category):
    """Insert or update a record in the quarterly_revenue table."""
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO quarterly_revenue (
        ticker, name, quarter, revenue, earnings_date, revenue_growth, revenue_average,
        gross_profit, net_income, ebitda, employee_count, trailing_pe, category
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(ticker, quarter) DO UPDATE SET
        revenue = excluded.revenue,
        earnings_date = excluded.earnings_date,
        revenue_growth = excluded.revenue_growth,
        revenue_average = excluded.revenue_average,
        gross_profit = excluded.gross_profit,
        net_income = excluded.net_income,
        ebitda = excluded.ebitda,
        employee_count = excluded.employee_count,
        trailing_pe = excluded.trailing_pe
    ''', (ticker, name, quarter, revenue, earnings_date, revenue_growth, revenue_average,
          gross_profit, net_income, ebitda, employee_count, trailing_pe, category))
    conn.commit()

def populate_initial_data(db_path):
    """Populate the database with initial data from the scraped tickers."""
    conn = sqlite3.connect(db_path)

    # Load tickers from the JSON file
    with open('tickers.json', 'r') as f:
        tickers = json.load(f)

    for ticker_data in tickers:
        ticker = ticker_data['ticker']
        category = ticker_data['category']

        try:
            stock = yf.Ticker(ticker)
            earnings_calendar = stock.calendar
            next_earnings_date = earnings_calendar.get('Earnings Date')
            if next_earnings_date is not None:
                next_earnings_date = pd.to_datetime(next_earnings_date[0]).date()
                latest_revenue = stock.quarterly_financials.loc['Total Revenue'].max() if 'Total Revenue' in stock.quarterly_financials.index else None
                revenue_growth = stock.quarterly_financials.loc['Total Revenue'].pct_change().mean() * 100 if 'Total Revenue' in stock.quarterly_financials.index else None
                revenue_average = None
                if isinstance(earnings_calendar, dict) and 'Revenue Average' in earnings_calendar:
                    revenue_average = earnings_calendar['Revenue Average']

                # Get other financial metrics
                quarterly_financials = stock.quarterly_financials
                gross_profit = quarterly_financials.loc['Gross Profit'].max() if 'Gross Profit' in quarterly_financials.index else None
                net_income = quarterly_financials.loc['Net Income'].max() if 'Net Income' in quarterly_financials.index else None
                ebitda = quarterly_financials.loc['EBITDA'].max() if 'EBITDA' in quarterly_financials.index else None
                employee_count = stock.info.get('fullTimeEmployees', None)
                trailing_pe = stock.info.get('trailingPE', None)

                # Determine the quarter of the next earnings date
                quarter = next_earnings_date.strftime('%Q-%Y')
                insert_quarterly_revenue(conn, ticker, stock.info.get('longName', 'N/A'), quarter, latest_revenue, next_earnings_date.strftime('%Y-%m-%d'), revenue_growth, revenue_average, gross_profit, net_income, ebitda, employee_count, trailing_pe, category)
        except Exception as e:
            logging.error("Error saving data for %s: %s", ticker, e)
            print(f"Error saving data for {ticker}: {e}")  # Debug statement

    conn.close()

def update_database(db_path):
    """Fetch data from Yahoo Finance and update the database."""
    conn = sqlite3.connect(db_path)
    tickers = load_tickers_from_db(db_path)

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            earnings_calendar = stock.calendar
            next_earnings_date = earnings_calendar.get('Earnings Date')
            if next_earnings_date is not None:
                next_earnings_date = pd.to_datetime(next_earnings_date[0]).date()
                latest_revenue = stock.quarterly_financials.loc['Total Revenue'].max() if 'Total Revenue' in stock.quarterly_financials.index else None
                revenue_growth = stock.quarterly_financials.loc['Total Revenue'].pct_change().mean() * 100 if 'Total Revenue' in stock.quarterly_financials.index else None
                revenue_average = None
                if isinstance(earnings_calendar, dict) and 'Revenue Average' in earnings_calendar:
                    revenue_average = earnings_calendar['Revenue Average']

                # Get other financial metrics
                quarterly_financials = stock.quarterly_financials
                gross_profit = quarterly_financials.loc['Gross Profit'].max() if 'Gross Profit' in quarterly_financials.index else None
                net_income = quarterly_financials.loc['Net Income'].max() if 'Net Income' in quarterly_financials.index else None
                ebitda = quarterly_financials.loc['EBITDA'].max() if 'EBITDA' in quarterly_financials.index else None
                employee_count = stock.info.get('fullTimeEmployees', None)
                trailing_pe = stock.info.get('trailingPE', None)

                # Determine the quarter of the next earnings date
                quarter = next_earnings_date.strftime('%Q-%Y')
                insert_quarterly_revenue(conn, ticker, stock.info.get('longName', 'N/A'), quarter, latest_revenue, next_earnings_date.strftime('%Y-%m-%d'), revenue_growth, revenue_average, gross_profit, net_income, ebitda, employee_count, trailing_pe)
        except Exception as e:
            logging.error("Error saving data for %s: %s", ticker, e)
            print(f"Error saving data for {ticker}: {e}")  # Debug statement

    conn.close()

# Initialize the database and populate it with data
db_path = 'financial_data.db'
initialize_database(db_path)

# Populate the database with initial data if it's empty
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM quarterly_revenue')
count = cursor.fetchone()[0]
conn.close()

if count == 0:
    populate_initial_data(db_path)

# --- Fetching Data Functions ---

def fetch_companies_with_earnings(db_path, ticker=None):
    """Fetch companies with earnings calls in the next week, optionally filtering by ticker."""
    conn = sqlite3.connect(db_path)
    if ticker:
        query = '''
            SELECT ticker, name, revenue, earnings_date, revenue_growth, revenue_average, 
                    gross_profit, net_income, ebitda, employee_count, trailing_pe, quarter
            FROM quarterly_revenue
            WHERE ticker = ? AND earnings_date >= DATE('now') AND earnings_date <= DATE('now', '+7 days')
        '''
        df = pd.read_sql_query(query, conn, params=(ticker,))
    else:
        query = '''
            SELECT ticker, name, quarter, revenue, earnings_date, revenue_growth, revenue_average, 
                    gross_profit, net_income, ebitda, employee_count, trailing_pe 
            FROM quarterly_revenue
            WHERE earnings_date >= DATE('now') AND earnings_date <= DATE('now', '+7 days')
        '''
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    # Filter out rows with NaN values in 'revenue', 'revenue_growth'
    df = df.dropna(subset=['revenue', 'revenue_growth'])
    return df

def check_earnings_for_ticker(db_path, ticker):
    """Check if a specific ticker has an earnings call in the next week."""
    conn = sqlite3.connect(db_path)
    query = '''
        SELECT ticker, name, quarter, revenue, earnings_date, revenue_growth, revenue_average, 
                gross_profit, net_income, ebitda, employee_count, trailing_pe
        FROM quarterly_revenue
        WHERE ticker = ? AND earnings_date >= DATE('now') AND earnings_date <= DATE('now', '+7 days')
    '''
    df = pd.read_sql_query(query, conn, params=(ticker,))
    conn.close()
    # Filter out rows with NaN values in 'revenue', 'revenue_growth'
    df = df.dropna(subset=['revenue', 'revenue_growth'])
    if not df.empty:
        earnings_date = pd.to_datetime(df['earnings_date'].values[0])
        return f"The next earnings call for {ticker} is on {earnings_date.strftime('%Y-%m-%d')}."
    else:
        return f"No earnings calls for {ticker} in the next 7 days."


# --- Streamlit Web App ---

# Function to load tickers from the database
def load_tickers_from_db(db_path):
    conn = sqlite3.connect(db_path)
    tickers_df = pd.read_sql_query("SELECT DISTINCT ticker FROM quarterly_revenue", conn)
    conn.close()
    return tickers_df['ticker'].tolist()

# Function to load distinct categories from the database
def load_categories_from_db(db_path):
    conn = sqlite3.connect(db_path)
    categories_df = pd.read_sql_query("SELECT DISTINCT category FROM quarterly_revenue", conn)
    conn.close()
    return categories_df['category'].dropna().tolist()

# Function to fetch upcoming earnings based on a custom date range
def fetch_upcoming_earnings(db_path, start_days, end_days, selected_category=None):
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT ticker, name AS 'Name', earnings_date AS 'Next Earnings Date', revenue AS 'Quarterly Revenue',
               revenue_growth AS 'Quarterly Revenue Growth (%)', revenue_average AS 'Next Revenue Estimate'
        FROM quarterly_revenue
        WHERE earnings_date BETWEEN DATE('now', '+{start_days} days') AND DATE('now', '+{end_days} days')
    """

    # Filter by selected category if one is chosen
    if selected_category:
        query += f" AND category = '{selected_category}'"

    query += " ORDER BY earnings_date"

    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Function to get company details from the database
def get_company_details(ticker, db_path):
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT name AS 'Name', earnings_date AS 'Next Earnings Date', revenue AS 'Quarterly Revenue',
               revenue_growth AS 'Quarterly Revenue Growth (%)', revenue_average AS 'Next Revenue Estimate',
               gross_profit AS 'Gross Profit', ebitda AS 'EBITDA', net_income AS 'Net Income',
               employee_count AS 'Employee Count', trailing_pe AS 'Trailing P/E', category AS 'Category'
        FROM quarterly_revenue
        WHERE ticker = '{ticker}'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Function to determine if the earnings beat or missed estimates
def get_earnings_beat_or_miss(ticker, db_path):
    conn = sqlite3.connect(db_path)
    query = f"SELECT revenue, revenue_average FROM quarterly_revenue WHERE ticker = '{ticker}'"
    result = pd.read_sql_query(query, conn)
    conn.close()

    if not result.empty:
        latest_revenue = result['revenue'].iloc[0]
        revenue_average = result['revenue_average'].iloc[0]

        if pd.isna(latest_revenue) or pd.isna(revenue_average):
            return 'Unknown'

        if latest_revenue > revenue_average:
            return 'Beat'
        elif latest_revenue < revenue_average:
            return 'Miss'
        else:
            return 'Met'
    else:
        return 'Unknown'
        
# Set the page title and other configurations
st.set_page_config(
    page_title="Earnings Checker App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display an informational introduction
st.markdown("""
##  Earnings Checker App

This app provides an overview of upcoming earnings calls for consumer companies. Using the filters in the sidebar, you can:

- **View Upcoming Earnings**: See which consumer companies have earnings calls scheduled in the next four-week period.
- **Filter by Company Category**: Select the consumer company category to view specific types of companies.
- **Filter by Earnings Performance**: View companies that beat or missed their earnings estimates.
- **Get Deep Dives**: Enter a specific ticker symbol to view detailed financial information for that company.

""")

# Set up the sidebar
st.sidebar.header('Filters')

# Add the slider for selecting the date range
st.sidebar.subheader('Earnings in the Next 28 Days')

date_range = st.sidebar.slider(
    'Select Date Range for Earnings:',
    min_value=0,  # Changed min_value to 0 to include today
    max_value=28,
    value=(0, 7)  # Default range starts from today
)

# Extract start and end days from the slider
start_days, end_days = date_range

# Select Consumer Company Category Section
st.sidebar.subheader('Filter for Company Category')
selected_category = st.sidebar.selectbox(
    'Category:',
    options=['All'] + load_categories_from_db('financial_data.db')
)

# View Hitters & Missers Section
st.sidebar.subheader('View Hitters & Missers')
beat_or_miss = st.sidebar.selectbox(
    'Select Earnings Performance:',
    options=['All', 'Beat', 'Miss']
)

# Select a Ticker Section
st.sidebar.subheader('Company Deep Dive')
selected_ticker = st.sidebar.text_input('Enter Ticker Symbol:', 
                                        placeholder='e.g. AMZN')

# Load tickers from the database
db_path = 'financial_data.db'
tickers = load_tickers_from_db(db_path)

# Display upcoming earnings within the selected date range
st.title('Upcoming Earnings Calls')

if selected_category == 'All':
    df_upcoming_earnings = fetch_upcoming_earnings(db_path, start_days, end_days)
else:
    df_upcoming_earnings = fetch_upcoming_earnings(db_path, start_days, end_days, selected_category)

# Apply filters for beat or miss
if beat_or_miss != 'All':
    df_upcoming_earnings['Beat/Miss'] = df_upcoming_earnings['ticker'].apply(lambda x: get_earnings_beat_or_miss(x, db_path))
    df_upcoming_earnings = df_upcoming_earnings[df_upcoming_earnings['Beat/Miss'] == beat_or_miss]

if not df_upcoming_earnings.empty:
    df_upcoming_earnings = df_upcoming_earnings.rename(columns={
        'ticker': 'Ticker Symbol',
        'Next Earnings Date': 'Next Earnings Date',
        'Quarterly Revenue': 'Quarterly Revenue',
        'Quarterly Revenue Growth (%)': 'Quarterly Revenue Growth (%)',
        'Next Revenue Estimate': 'Next Revenue Estimate',
    })

    # Convert 'Next Earnings Date' to datetime
    df_upcoming_earnings['Next Earnings Date'] = pd.to_datetime(df_upcoming_earnings['Next Earnings Date']).dt.date

    # Apply formatting and highlighting
    styled_df = df_upcoming_earnings.style.format({
        'Quarterly Revenue': '${:,.0f}',
        'Quarterly Revenue Growth (%)': '{:.0f}%',
        'Next Revenue Estimate': '${:,.0f}'
    })

    # Display the DataFrame with custom width and hiding index
    st.dataframe(styled_df, hide_index=True, use_container_width=True)
else:
    st.info(f"No companies with earnings calls in the next {end_days} days.")

# Only display the detailed table if a ticker is selected
if selected_ticker:
    st.subheader(f"Earnings details for {selected_ticker}")

    company_details = get_company_details(selected_ticker, db_path)

    if not company_details.empty:
        beat_or_miss_status = get_earnings_beat_or_miss(selected_ticker, db_path)
        st.write(f"Earnings Performance in the Previous Quarter: **{beat_or_miss_status}**")

        # Transpose the DataFrame
        transposed_details = company_details.transpose()

        # Add a 'Metric' column to indicate what each value represents
        transposed_details.reset_index(inplace=True)
        transposed_details.columns = ['Metric', 'Value']

        # Apply formatting to USD values and percentages
        def format_value(val, metric):
            if isinstance(val, (int, float)):
                # Check for Growth or P/E first to apply percentage formatting
                if 'Growth' in metric or 'P/E' in metric:
                    return f"{val:.0f}%"
                # Then check for Revenue, Profit, or related metrics to apply currency formatting
                elif any (keyword in metric for keyword in ['Revenue', 'Profit', 'EBITDA', 'Net Income']):
                    return f"${val:,.0f}"
            return val

        # Use apply with lambda to pass both value and metric
        transposed_details['Value'] = transposed_details.apply(lambda x: format_value(x['Value'], x['Metric']), axis=1)
        # Display the transposed DataFrame with custom formatting and hide the index
        st.dataframe(transposed_details.style.hide(axis="index"))
    else:
        st.warning("This database only covers companies in the consumer category.")

# Apply CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .dataframe thead th {
        background-color: #007BFF;
        color: white;
    }
    .dataframe tbody tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    .dataframe td, .dataframe th {
        padding: 5px;
        max-width: 300px;
        text-overflow: ellipsis;
        overflow: hidden;
        white-space: nowrap;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #007BFF;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
**Note:** This app gets data from Yahoo Finance. Some metrics may display as "None" or be missing if the company has not made this information publicly available. The earnings data provided here is for informational purposes only. Always consult official financial reports and disclosures before making any investment decisions.
""")

