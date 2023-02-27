import json


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


DEFAULT_CONFIGURATION_PATH = 'config//default_configuration.txt'
CURRENT_CONFIGURATION_PATH = 'config//current_configuration.txt'


class Configurations(metaclass=Singleton):

    def __init__(self):
        self.default_configuration = self.load_configuration(DEFAULT_CONFIGURATION_PATH)
        self.current_configuration = self.load_configuration(CURRENT_CONFIGURATION_PATH)

    @staticmethod
    def load_configuration(path):
        with open(path) as configuration_file:
            return json.loads(configuration_file.read())

    def change_current_configuration(self, configuration_data):
        self.current_configuration = configuration_data
        with open('config//current_configuration.txt', 'w') as f:
            print(json.dumps(configuration_data), file=f)
