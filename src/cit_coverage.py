import tempfile
import os
import coverage
import importlib.util
import uuid
from types import ModuleType
from typing import Callable, List, Any

def get_coverage(code: str, func_name: str, func_kwargs: dict = {}) -> List[int]:
    """
    Executes a function from a given code snippet and returns a list of line numbers that were executed.

    Parameters:
    - code (str): The code of the module containing the function.
    - func_name (str): The name of the function to test.
    - func_args (tuple): Positional arguments to pass to the function.
    - func_kwargs (dict): Keyword arguments to pass to the function.

    Returns:
    - List[int]: A list of executed line numbers within the code snippet.
    """

    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    mod_name = f"temp_mod_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, tmp_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cov = coverage.Coverage()
    cov.start()

    try:
        func: Callable = getattr(mod, func_name)
        func(**func_kwargs)
    finally:
        cov.stop()
        cov.save()

    # Analyze the coverage
    analysis = cov.analysis(tmp_path)
    executed_lines = analysis[1]  # line numbers executed
    cov.erase()

    # Clean up
    os.remove(tmp_path)

    return executed_lines
