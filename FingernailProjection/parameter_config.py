from dataclasses import dataclass
from panel import *
import param
import json
from typing import Dict, List, Type, TypeVar
from enum import Enum
from helper import join_string
from enums import *
from tesseract_enums import *
import panel as pn


T = TypeVar("T", bound="ParameterConfig")


@dataclass
class Category:
    name: str
    parameters: List[param.Parameter]
    expanded: bool


class ParameterConfig(param.Parameterized):
    """Base class for saving and loading param settings."""

    def __init__(self, **params):
        super().__init__(**params)
        self.path = ""

    def load_from_file(self, json_filepath: str) -> bool:
        """Loads the configuration from a JSON file."""
        try:
            changed_attributes_names = []
            with open(json_filepath, "r") as file:
                data = json.load(file)

            for attribute_name, value in data.items():
                if attribute_name == "name":
                    continue

                if hasattr(self, attribute_name):
                    param_obj = getattr(self.param, attribute_name)
                    if isinstance(param_obj.default, Enum):
                        # Handle Enum parameters
                        try:
                            value_to_set = param_obj.default.__class__[value]
                        except KeyError as e:
                            print(f"Error setting Enum parameter: {e}")
                            continue
                    else:
                        value_to_set = value
                    previous_value = getattr(self, attribute_name)
                    if previous_value != value_to_set:
                        changed_attributes_names.append(attribute_name)
                    setattr(self, attribute_name, value_to_set)
            print(f"Successfully loaded config file: {json_filepath}")
            print(f"Changed attributes: {join_string(changed_attributes_names)}")
            self.path = json_filepath
            return True
        except Exception as e:
            print(f"Error loading config file: {e}\n{json_filepath}")
            return False

    def save_to_file(self, json_filepath: str):
        """Saves the current configuration to a file in JSON format."""
        data = {}
        for name in self.param:
            if name == "name":
                continue
            value = getattr(self, name)
            # Convert Enums to their names for serialization
            if isinstance(value, Enum):
                data[name] = value.name
            else:
                data[name] = value
        with open(json_filepath, "w") as file:
            json.dump(data, file, indent=4)

    def infer_categories(self) -> List[Category]:
        """Infer categories based on `category_` attributes in the class."""
        categories = []  # Initialize an empty list to store inferred categories
        current_category_name = ""  # Variable to track the current category name

        # Iterate through attributes of the class type using vars
        for attribute_name, enabled in vars(type(self)).items():
            if attribute_name == "name":
                continue  # Skip the "name" attribute

            if attribute_name.startswith("category_"):
                # Extract the category name from the attribute
                raw_category_name = attribute_name[len("category_") :]
                current_category_name = raw_category_name.replace("_", " ").title()
                # Add a new Category object with an empty parameter list
                categories.append(Category(current_category_name, [], enabled))

            elif attribute_name in self.param:
                # If the attribute is a parameter, assign it to the current category
                if categories:  # Ensure there is an existing category to assign to
                    categories[-1].parameters.append(getattr(self.param, attribute_name))

        return categories  # Return the list of inferred categories

    def add_config_changed_callback(self, config_changed: callable):
        """Watch all parameters for changes and call the callback."""
        self.param.watch(lambda _: config_changed(), list(self.param))

    def __init__(self):
        super().__init__()
        self.add_config_changed_callback(lambda: self.update_visibility())

    def make_all_invisible(self):
        for param in self.param.objects("existing").values():
            param.precedence = -1

    def make_all_visible(self):
        for param in self.param.objects("existing").values():
            param.precedence = 0

    def update_visibility(self, set_conditioned_params_false: bool = False):
        for param in self.param.objects("existing").values():
            if not param.doc:
                continue

            if condition := (
                AndCondition.dedoc(param.doc) if "and" in param.doc else Condition.dedoc(param.doc)
            ):
                # At the beginning, we need to set all parameters that COULD become
                # hidden actually to hidden. Else, they can't be hidden later on. :/
                if set_conditioned_params_false:
                    param.precedence = -1
                    continue

                param.precedence = 0 if condition.is_enabled(self) else -1
                # print(
                #     f"{name} enabled = {is_enabled} because {enabled_if.param_name} {enabled_if.get_actual_value(self)} = {enabled_if.set_value}"
                # )
            elif "String identifier for this object." not in param.doc:
                print(f"Error parsing condition: {param.doc}")

    def load_accordion(self, path):
        # Changes will trigger a file save
        self.add_config_changed_callback(lambda: self.save_to_file(path))

        self.update_visibility(True)

        # Generate category panels and display them in tabs
        pn.extension()
        categories = self.infer_categories()
        accordion = pn.Accordion(
            *[(category.name, pn.Column(*category.parameters)) for category in categories]
        )

        # Set only enabled categories as active
        accordion.active = [index for index, category in enumerate(categories) if category.expanded]
        print("Created accordion")

        # Update visibility when the page is loaded
        pn.state.onload(lambda: self.update_visibility())

        return accordion

    def load_accordion_from_file(self, path):
        if not self.load_from_file(path):
            print("Error loading config from path " + path)
        return self.load_accordion(path)


class Condition:
    """Enables or disables a parameter based on the value of another parameter."""

    def __init__(self, param_name_to_check: str, set_value: bool):
        self.param_name = param_name_to_check
        self.set_value = set_value

    def get_actual_value(self, config: ParameterConfig):
        return getattr(config, self.param_name, None)

    def is_enabled(self, config: ParameterConfig):
        return str(self.get_actual_value(config)) == str(self.set_value)

    @property
    def doc(self) -> str:
        """Encodes the field name and value into a doc string."""
        return f"{self.param_name}={self.set_value}"

    def dedoc(doc: str) -> "Condition":
        """Decodes the field name and value from a doc string."""
        splits = doc.split("=")
        if len(splits) == 2:
            return Condition(splits[0], splits[1])
        return None

    def __and__(self, other: "Condition") -> "AndCondition":
        return AndCondition([self, other])


class AndCondition(Condition):
    """Combines multiple conditions into one."""

    def __init__(self, conditions: list):
        self.conditions = conditions

    def is_enabled(self, config: ParameterConfig):
        for condition in self.conditions:
            if not condition.is_enabled(config):
                return False
        return True

    @property
    def doc(self) -> str:
        return join_string([condition.doc for condition in self.conditions], " and ")

    @staticmethod
    def dedoc(doc: str) -> "AndCondition":
        """Decodes multiple conditions from a combined doc string."""
        condition_docs = doc.split(" and ")
        conditions = [Condition.dedoc(cond_doc) for cond_doc in condition_docs]
        return AndCondition(conditions)
