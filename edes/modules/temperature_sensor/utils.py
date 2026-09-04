import requests

def query_temp(stage, T_to_channels = {'40K': 1, '4K': 2, '0.4K': 5}): 
    req = requests.get('http://192.168.169.102:5001/channel/measurement/latest', timeout=10)
    data = req.json()
    ch = T_to_channels[stage]
    while data['channel_nr'] != ch: 
        req = requests.get('http://192.168.169.102:5001/channel/measurement/latest', timeout=10)
        data = req.json()
    return data['temperature']