def get_coverage(test):
    # Simulate coverage as a set of features based on parameter values
    return {f"{param}={value}" for param, value in test.items()}

def intersect_coverage(coverage_list):
    if not coverage_list:
        return set()
    return set.intersection(*coverage_list)
