from file_watcher import FileWatcher
from typing import Type, TypeVar, Generic, Tuple
from parameter_config import ParameterConfig

# Define a type variable for classes derived from ParameterConfig
TConfig = TypeVar("TConfig", bound="ParameterConfig")


class AutomaticConfigReloader(Generic[TConfig]):
    """Facilitate the usage of a generic Config class by providing a file watcher and automatic reloading."""

    def __init__(self, path: str, config_cls: Type[TConfig]):
        self.path = path
        self.config = config_cls()  # Initialize the config instance
        self.config.load_from_file(path)  # Load configuration from the specified file
        self.config_file_watcher = FileWatcher(
            path
        )  # Initialize a file watcher for the config file

    def reload_if_config_changed(self):
        """Reload the configuration if the file timestamp has changed."""
        if self.config_file_watcher.has_timestamp_changed():
            self.config.load_from_file(self.path)

    @staticmethod
    def get_config_and_reloader(
        path: str, config_cls: Type[TConfig]
    ) -> Tuple[TConfig, "AutomaticConfigReloader[TConfig]"]:
        """
        Create and return a config instance and an AutomaticConfigReloader.
        """
        reloader = AutomaticConfigReloader(path, config_cls)
        return reloader.config, reloader

    @staticmethod
    def get_loader(path: str, config_cls: Type[TConfig]) -> "AutomaticConfigReloader[TConfig]":
        """
        Create and return an AutomaticConfigReloader.
        """
        return AutomaticConfigReloader(path, config_cls)
