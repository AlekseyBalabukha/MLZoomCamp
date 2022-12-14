import numpy as np
from locust import task
from locust import between
<<<<<<< HEAD
from locust import HttpUser
=======
from locust import HttpUser

#Sample data to send:
sample = {
    "seniority": 3,
    "home": "owner",
    "time": 36,
    "age": 26,
    "marital": "single",
    "records": "no",
    "job": "freelance",
    "expenses": 35,
    "income": 0.0,
    "assets": 60000.0,
    "debt": 3000.0,
    "amount": 800,
    "price": 1000
}

# inherit HttpUser object from locust:
class CreditRiskTestUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:
            locust -H http://localhost:3000, in case if all requests failed then load client with
            locust -H http://localhost:3000 -f locustfile.py

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming
    """

    # create method with task decorator to send request
    @task
    def classify(self):
        self.client.post("/classify", json=sample) # post request in json format with the endpoint 'classify'

    wait_time = between(0.01, 2) # set random wait time
>>>>>>> 7c9f85439b5df0dbf0199f53e183c3dac6ce8f7b
