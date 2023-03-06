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

    def read(self, path, configuration=None, backup_configuration=None):
        if configuration is None:
            configuration = self.current_configuration
        if backup_configuration is None:
            backup_configuration = self.default_configuration

        split_path = path.split('.')
        key = split_path[0]

        value = None
        backup_value = None
        try:
            value = configuration[key]
        except KeyError:
            value = None
            try:
                backup_value = backup_configuration[key]
            except KeyError:
                backup_value = None
        if value:
            if len(split_path) == 1:
                return self.parse_value(value)
            return self.read('.'.join(split_path[1:]), value, backup_value)
        elif backup_value:
            if len(split_path) == 1:
                return self.parse_value(backup_value)
            return self.read('.'.join(split_path[1:]), backup_value, backup_value)
        return None

    @staticmethod
    def load_configuration(path):
        with open(path) as configuration_file:
            return json.loads(configuration_file.read())

    @staticmethod
    def parse_value(element):
        if element['type'] == 'list':
            return element['value']
        if element['type'] == 'int':
            return int(element['value'])

    def change_current_configuration(self, configuration_data):
        self.current_configuration = configuration_data
        with open('config//current_configuration.txt', 'w') as f:
            print(json.dumps(configuration_data), file=f)
