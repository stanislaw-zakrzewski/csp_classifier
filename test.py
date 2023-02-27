import json
import time
from config.config import Configurations

with open('config/default_configuration.txt', 'w') as f:
    print(json.dumps({'sampling_rate': [250,'250']}), file=f)

with open('config/default_configuration.txt') as f:
    data = f.read()

js = json.loads(data)
print(js)
print(type(js['sampling_rate']))

c = Configurations()
print(c.t)
time.sleep(3)
c2 = Configurations()
print(c2.t)