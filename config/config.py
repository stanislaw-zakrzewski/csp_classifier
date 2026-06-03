import json
from json.decoder import JSONDecodeError
import os


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


DEFAULT_CONFIGURATION_PATH = os.path.join('config', 'default_configuration.json')
CURRENT_CONFIGURATION_PATH = os.path.join('config', 'current_configuration.json')


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
        except (KeyError, TypeError):
            value = None
            
        try:
            backup_value = backup_configuration[key]
        except (KeyError, TypeError):
            backup_value = None

        if value is not None:
            if len(split_path) == 1:
                return value
            return self.read('.'.join(split_path[1:]), value, backup_value)
        elif backup_value is not None:
            if len(split_path) == 1:
                return backup_value
            return self.read('.'.join(split_path[1:]), backup_value, backup_value)
        return None

    @staticmethod
    def load_configuration(path):
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as configuration_file:
            try:
                return json.load(configuration_file)
            except (JSONDecodeError, IOError):
                return {}

    def change_current_configuration(self, configuration_data):
        self.current_configuration = configuration_data
        with open(CURRENT_CONFIGURATION_PATH, 'w', encoding='utf-8') as f:
            json.dump(configuration_data, f, indent=2)
