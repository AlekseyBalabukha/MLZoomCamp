import requests
# host AWS:
host = 'churn-serving-env.eba-ypmmdpnx.eu-west-1.elasticbeanstalk.com'
url = f'http://{host}/predict' # we do not need port, since AWS maps to 80
# customer in json format:
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}

# making POST request to the Predict Web Service:
response = requests.post(url, json=customer).json()
if response['churn'] == True:
    print(f'sending promo email to customer: 999')