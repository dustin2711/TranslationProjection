from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ConfigSetting(Generic[T]):
    display_name: str
    default_value: T


@dataclass
class ConfigSettingFloat(ConfigSetting[int]):
    min_value: int
    max_value: int


def config_setting(cls):
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        # Initialize any original init behavior
        original_init(self, *args, **kwargs)

        # Assign default values as instance attributes
        for name, obj in cls.__dict__.items():
            if isinstance(obj, ConfigSetting):
                obj.variable_name = name  # Set variable_name to the attribute name
                setattr(self, name, obj.default_value)

    cls.__init__ = new_init
    return cls


@config_setting
class Config:
    # Define settings directly without specifying variable_name
    slider_enabled = ConfigSetting("Slider enabled", False)
    floatslider = ConfigSettingFloat("Float Slider", 7, 0, 100)

    @staticmethod
    def get_setting(variable_name):
        # Retrieve the ConfigSetting object based on the variable name
        for name, obj in Config.__dict__.items():
            if isinstance(obj, ConfigSetting) and obj.variable_name == variable_name:
                return obj
        return None


# Testing the setup
config = Config()
config.stringtext = "changed string"

# Retrieve the ConfigSetting object and access metadata
setting = Config.get_setting("floatslider")
if setting:
    print(f"Display Name: {setting.display_name}")  # Output: "Float Slider"
    print(f"Variable Name: {setting.variable_name}")  # Output: "floatslider"
    print(f"Default Value: {setting.default_value}")  # Output: 7
