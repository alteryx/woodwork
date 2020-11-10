from .inference import *

DEFAULT_INFERENCE_FUNCTIONS = {
    'Categorical': categorical_func,
    'Integer': integer_func,
    'WholeNumber': whole_number_func,
    'Double': double_func,
    'Boolean': boolean_func,
    'Datetime': datetime_func,
    'Timedelta': timedelta_func,
}

CONFIG_DEFAULTS = {
    'natural_language_threshold': 10,
    'numeric_categorical_threshold': -1,
    'inference_functions': DEFAULT_INFERENCE_FUNCTIONS
}


class Config:
    def __init__(self, default_values):
        self._defaults = default_values
        self._data = default_values.copy()

    def set_option(self, key, value):
        if key not in self._data.keys():
            raise KeyError(f"Invalid option specified: {key}")
        self._data[key] = value

    def get_option(self, key):
        if key not in self._data.keys():
            raise KeyError(f"Invalid option specified: {key}")
        return self._data[key]

    def reset_option(self, key):
        if key not in self._data.keys():
            raise KeyError(f"Invalid option specified: {key}")
        self._data[key] = self._defaults[key]

    def set_inference_function(self, logical_type, inference_function):
        inference_functions = self._data['inference_functions'].copy()
        inference_functions[logical_type] = inference_function
        self.set_option('inference_functions', inference_functions)

    def __repr__(self):
        output_string = "Woodwork Global Config Settings\n"
        output_string += "-" * (len(output_string) - 1)
        for key, value in self._data.items():
            output_string += f"\n{key}: {value}"
        return output_string


config = Config(CONFIG_DEFAULTS)
