"""Utility decorators for function validation."""

import inspect
import re
from functools import wraps

import xarray as xr


def check_docstring_dims(func):
    """Decorator to check if the dimensions of xr.DataArray arguments and return values
    match the docstring specification.

    This decorator validates that:
    - Input xr.DataArray arguments have dimensions matching the docstring
    - Return xr.DataArray has dimensions matching the docstring

    Parameters
    ----------
    func : callable
        Function to decorate. Should have dimension specifications in its docstring.

    Returns
    -------
    callable
        Wrapped function with dimension validation.

    Raises
    ------
    AssertionError
        If dimensions don't match the docstring specification.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Retrieve function signature and bound arguments
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Retrieve docstring
        docstring = inspect.getdoc(func)
        if not docstring:
            return func(*args, **kwargs)  # Skip check if no docstring

        # Extract expected input dimensions
        param_pattern = re.compile(r"(\w+)\s*:\s*xr\.DataArray\s*\((.*?)\)")
        expected_dims = {
            match[0]: tuple(match[1].split(", "))
            for match in param_pattern.findall(docstring)
        }

        # Validate input arguments
        for arg_name, expected_dim in expected_dims.items():
            if arg_name in bound_args.arguments:
                arg_value = bound_args.arguments[arg_name]
                if isinstance(arg_value, xr.DataArray):
                    actual_dim = arg_value.dims
                    assert (
                        actual_dim == expected_dim
                    ), f"Argument '{arg_name}' expected dimensions {expected_dim}, "
                    f"but got {actual_dim}."

        # Execute the function
        result = func(*args, **kwargs)

        # Extract expected output dimensions
        return_pattern = re.compile(
            r"Returns\s*\n[-]+\n\s*(\w+)\s*:\s*xr\.DataArray\s*\((.*?)\)"
        )
        match = return_pattern.search(docstring)
        if match:
            return_var, expected_return_dim = match.groups()
            expected_return_dim = tuple(expected_return_dim.split(", "))

            # Validate return value
            if isinstance(result, xr.DataArray):
                actual_return_dim = result.dims
                assert actual_return_dim == expected_return_dim, (
                    f"Function '{func.__name__}' "
                    f"expected return dimensions {expected_return_dim}, "
                    f"but got {actual_return_dim}."
                )

        return result  # Return function output as usual

    return wrapper
