import pandas as pd 
def get_data():
  data = {
    "location": ["New York", "Los Angeles", "Chicago", "Houston", "Miami"],
    "price_per_sqm": [10000, 7500, 5000, 4000, 6000]  
  }
  return pd.DataFrame(data)