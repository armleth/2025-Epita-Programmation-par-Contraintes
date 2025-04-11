import coverage
import unittest
import sys
from io import StringIO


def get_coverage(test_suite, strategy_name):
    # Create a string buffer to capture output
    captured_output = StringIO()
    
    # Redirect stdout to our buffer
    original_stdout = sys.stdout
    sys.stdout = captured_output
    
    runner = unittest.TextTestRunner(verbosity=0)
    cov = coverage.Coverage(
        # Exclude testsuite.py and other test files
        omit=['testsuite.py', '*/__init__.py'],
    )
    cov.start()
    try:
        runner.run(test_suite)
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        cov.stop()
        cov.save()

    data = cov.get_data()
    lines = []
    for file_name in data.measured_files():
        for line in data.lines(file_name):
            lines.append(line)

    return sorted(lines)
