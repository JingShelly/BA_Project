
#------------------Stock ETFs------------------
stock_etfs = {
    "SPY"      : "Tracks the S&P 500 Index, providing exposure to large-cap U.S. companies.",
    "ASHR.L"   : "Tracks the performance of the FTSE China A50 Index, providing exposure to the 50 largest A-share companies in China.",
    "VPL"      : "Tracks the FTSE Developed Asia Pacific Index, providing exposure to developed markets in the Asia-Pacific region.",
    "INDA"     : "Tracks the performance of the MSCI India Index, providing exposure to Indian equities.",
}

stock = list(stock_etfs.keys())

#------------------Bond ETFs------------------
bond_etfs = {
    "TLT": "Tracks long-term U.S. Treasury bonds, focusing on maturities greater than 20 years.",
}
bond = list(bond_etfs.keys())

#------------------Emerging Market ETFs------------------
markets_dict = {
    "EMXC": "China",
    "IEMG": "Tracks emerging market countries such as China, India, Brazil, and South Korea.",
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
    "AIEQ"   : "Tracks the performance of AI-powered investment strategy, using machine learning to select stocks.",
}
AI = list(AI_etfs.keys())

#------------------Crypto ETFs------------------
crypto_etfs = {
    "BCHN.L"   : "Tracks the performance of Bitcoin Cash.",
    # "ETHW.L"   : "Tracks the performance of Ethereum.",
}
crypto = list(crypto_etfs.keys())

#------------------Commodity ETFs------------------
commodity_etfs = {
    "GLD": "Tracks the price of gold ",
    "SLV": "Tracks the price of silver ",
    "USO": "Tracks the price of crude oil in US.",
    "DBO": "Tracks the performance of crude oil futures contracts",
    "CORN": "Tracks the price of corn futures contracts"
}

commodity = list(commodity_etfs.keys())

defense_etfs = {
    "PSCC": "Tracks the performance of the S&P SmallCap 600 Consumer Staples Index",
}

defense = list(defense_etfs.keys())

#------------------Energy ETFs------------------
energy_etfs = {
    "USO": "crude oil",
    "VDE": "Tracks the performance of the energy sector of the U.S. stock market.",
}

energy = list(energy_etfs.keys())


#------------------Real Estate ETFs------------------
real_estate_etfs = {
    "IDUP.L": "Tracks the performance of the MSCI World Real Estate Index, providing exposure to real estate companies around the world.",
    "VNQ": "Tracks the performance of the MSCI US REIT Index, providing exposure to U.S. real estate companies.",
}

real_estate = list(real_estate_etfs.keys())

# Top 10 oldest ETFs
ten_oldest_etfs = {
    "EWU": "Tracks the performance of UK equities.",
    "EWS": "Tracks the performance of Singapore equities.",
    "EWW": "Tracks the performance of Mexican equities.",
    "EWJ": "Tracks the performance of Japanese equities.",
    "EWH": "Tracks the performance of Hong Kong equities.",
    "EWG": "Tracks the performance of German equities.",
    "EWQ": "Tracks the performance of French equities.",
    "EWC": "Tracks the performance of Canadian equities.",
    "EWA": "Tracks the performance of Australian equities.",
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
    "stock_etfs": stock,
    "bond_etfs": bond,
    "markets_dict": market,
    "technology_etfs": technology,
    "AI_etfs": AI,
    "crypto_etfs": crypto,
    "commodity_etfs": commodity,
    "defense_etfs": defense,
    "energy_etfs": energy,
    "real_estate_etfs": real_estate,
    "ten_oldest_etfs": ten_oldest
}
etf_category_map = {}

# assets = yf.download(etfs)
for category_name, etf_list in categories.items():
    for etf in etf_list:
        etf_category_map[etf] = category_name

# Print out the category mapping
# for etf, category in etf_category_map.items():
#     print(f"{etf} belongs to category: {category}")


# https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e

