
import yfinance as yf
from datetime import datetime, timezone
import pandas as pd

#------------------Stock ETFs------------------
stock_etfs = {
    "SPY"      : "S&P 500 Index",
    "ASHR.L"   : "China A50 Index",
    "VPL"      : "Developed Asia Pacific Index",
    "INDA"     : "India Index",
}

stock = list(stock_etfs.keys())

#------------------Bond ETFs------------------
bond_etfs = {
    "TLT": "long-term U.S. Treasury bonds",
}
bond = list(bond_etfs.keys())

#------------------Emerging Market ETFs------------------
markets_dict = {
    "EMXC": "China",
    "IEMG": "China, India, Brazil, and South Korea.",
    # "LYRIO.SW":"Brazil",
    "XMAF.L":"Africa",
    "VDNR.L": "North America",
    "KSA": "Saudi Arabia, Middle East",
    "EDEN": "Denmark",
    "VGK": "Europe",    
}

market = list(markets_dict.keys())


#------------------Technology ETFs------------------

technology_etfs = {
    # "SMH"    : "Tracks the performance of the **NYSE Arca Semiconductor Index**, providing exposure to companies in the semiconductor sector.",
    # "BTEC.SW": "BioTech",
    "SEMI.AS": "Semiconductors",
}

technology = list(technology_etfs.keys())

AI_etfs = {
    "AIEQ"   : "AI-powered investment strategy",
}
AI = list(AI_etfs.keys())

#------------------Crypto ETFs------------------
crypto_etfs = {
    "BCHN.L"   : "Bitcoin Cash.",
    # "ETHW.L"   : "Tracks the performance of Ethereum.",
}
crypto = list(crypto_etfs.keys())

#------------------Commodity ETFs------------------
commodity_etfs = {
    "GLD": "Gold ",
    "SLV": "SIlver ",
    "USO": "Crude oil in US.",
    "DBO": "Crude oil futures contracts",
    "CORN": "Corn futures contracts"
}

commodity = list(commodity_etfs.keys())

defense_etfs = {
    "PSCC": "S&P SmallCap 600 Consumer Staples",
}

defense = list(defense_etfs.keys())

#------------------Energy ETFs------------------
energy_etfs = {
    "USO": "crude oil",
    "VDE": "nergy sector of the U.S",
}

energy = list(energy_etfs.keys())


#------------------Real Estate ETFs------------------
real_estate_etfs = {
    "IDUP.L": "World Real Estate Index",
    "VNQ": "US real estate companies.",
}

real_estate = list(real_estate_etfs.keys())

# Top 10 oldest ETFs
ten_oldest_etfs = {
    "EWU": "UK equities.",
    "EWS": "Singapore equities.",
    "EWW": "Mexican equities.",
    "EWJ": "Japanese equities.",
    "EWH": "Hong Kong equities.",
    "EWG": "German equities.",
    "EWQ": "French equities.",
    "EWC": "Canadian equities.",
    "EWA": "Australian equities.",
    # "MDY": "Tracks the performance of the S&P MidCap 400 Index.",
    # "SPY": "Tracks the S&P 500 Index."
}
ten_oldest = list(ten_oldest_etfs.keys())


etfs_dicts = (stock_etfs | bond_etfs | markets_dict | technology_etfs | AI_etfs |
        crypto_etfs | commodity_etfs | defense_etfs | energy_etfs | 
        real_estate_etfs |ten_oldest_etfs)

etfs = list(etfs_dicts.keys())
total_assets = len(etfs)
print(f"Total number of ETFs: {total_assets}")

categories = {
    "Stock": stock,
    "Bond": bond,
    "Markets": market,
    "Technology": technology,
    "AI": AI,
    "Crypto": crypto,
    "Commodity": commodity,
    "Defense": defense,
    "Energy": energy,
    "Real Estate": real_estate,
    "Ten Oldest": ten_oldest
}
etf_category_map = {}

assets = yf.download(etfs)
for category_name, etf_list in categories.items():
    for etf in etf_list:
        etf_category_map[etf] = category_name

# Print out the category mapping
# for etf, category in etf_category_map.items():
#     print(f"{etf} belongs to category: {category}")

def get_first_date(etf_info):
    inception_date = etf_info.get("fundInceptionDate", None)
    if inception_date:
        if isinstance(inception_date, int):
            inception_date = datetime.fromtimestamp(inception_date, tz=timezone.utc)
        return inception_date.date()  
    return "N/A"


etf_data = []

for etf, comment in etfs_dicts.items():
    etf_info = yf.Ticker(etf).info
    category = etf_category_map[etf]
    inception_date = get_first_date(etf_info)
    currency = etf_info.get("currency", "Unknown")
    total_assets = etf_info.get("totalAssets", "N/A")


    etf_data.append([category,
                     etf,
                     inception_date,
                     currency,
                     total_assets,
                     comment])


df = pd.DataFrame(etf_data, columns=["Category",
                                     "ETF Name", 
                                     "Inception Date",
                                     "Currency",
                                     "Total Assets",
                                     "Comment"])

df['Total Assets'] = pd.to_numeric(df['Total Assets'], errors='coerce')
df["Total Assets"] = df["Total Assets"].apply(lambda x: f"{x:.5e}" if pd.notnull(x) else x)


# ------------------Save data to CSV and Excel------------------
df.to_csv('Final/ETFs/etf_summary.csv', index=False)
df.to_excel("Final/ETFs/etf_summary.xlsx", index=False)
print("The data has been saved to src/etf_summary.xlsx")

print(f"etf list: {etfs}")