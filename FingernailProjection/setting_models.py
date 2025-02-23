from dataclasses import dataclass
from enum import Enum


@dataclass
class SettingModel:
    display_name: str


@dataclass
class SettingModelCategory(SettingModel):
    pass


@dataclass
class UiSetting(SettingModel):
    variable_name: str
    value: int


@dataclass
class SettingModelSliderFloat(UiSetting):
    min_value: float = 0.0
    max_value: float = 1.0


@dataclass
class SettingModelSliderInt(SettingModelSliderFloat):
    min_value: int = 0
    max_value: int = 255


@dataclass
class SettingModelCheckbox(UiSetting):
    pass


@dataclass
class SettingModelCombobox(UiSetting):
    def __post_init__(self):
        if isinstance(self.value, Enum):
            self.enum_values = list(self.value.__class__)
        else:
            self.enum_values = []
