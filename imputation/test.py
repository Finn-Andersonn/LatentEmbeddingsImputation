import os
import requests
import json

# Ensure your API key is set properly
api_key = os.getenv("EIA_API_KEY")
if not api_key:
    api_key = "blgDYmYTswTZmly5SqN4eZ9qq4m5Z0DfQ8q7q5b8"  # Replace with your actual API key for testing

# Define the endpoint and parameters.
# Note: Using full series codes for daily prices.
base_url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"

params = {
    "frequency": "monthly",
    "data[0]": "value",
    "facets[series][]": ["PET.RWTC.D", "PET.RBRTE.D"],
    "start": "2014-01-18",
    "end": "2018-04-05",
    "sort[0][column]": "period",
    "sort[0][direction]": "asc",
    "offset": 0,
    "length": 5000,
    "api_key": api_key
}

# Make the request
response = requests.get(base_url, params=params)
# Print the full JSON response for debugging
print(json.dumps(response.json(), indent=2))
