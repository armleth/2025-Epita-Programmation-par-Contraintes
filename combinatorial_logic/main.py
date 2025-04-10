from uniform_t_way import * 
from mixed_t_way import *
from interaction_utils import *

params = [
    "Type",
    "Size",
    "Format method",
    "File system",
    "Cluster size",
    "Compression",
]

values = {
    "Type": [1, 2, 3],
    "Size": [1, 2, 3],
    "Format method": [1, 2],
    "File system": [1, 2, 3],
    "Cluster size": [1, 2, 3],
    "Compression": [1, 2],
}

# Constraints using index-based logic
def constraint1(v):
    return Implies(
        v["Type"] == values["Type"].index(2),
        v["Compression"] == values["Compression"].index(1),
    )

def constraint2(v):
    return Implies(
        v["Size"] >= values["Size"].index(1),
        v["Format method"] == values["Format method"].index(2),
    )

constraints = [constraint1, constraint2]

# Generate t-wise test suite (e.g., t = 2 for pairwise, t = 3 for 3-wise)
t = 2
test_suite = t_wise_testing(params, values, constraints, t=t)

locating_array = build_locating_array_matrix(test_suite, values)
pair_coverages = get_pair_coverages(locating_array, test_suite, values, display=False)
pair_impacts = get_pair_impact(pair_coverages, display=False)
param_impacts = get_param_impact(pair_coverages, display=False)
correlation_matrix = get_pair_correlation_matrix(param_impacts, pair_impacts, display=False)
strength_classes = strength_calculation(params, correlation_matrix, display=False)
unrolled_strength_classes =  define_new_combinations(strength_classes)
extended_test_suite = extend_test_suite(test_suite, values, constraints, unrolled_strength_classes)

# Print the extended test suite
for idx, test_case in enumerate(extended_test_suite):
    print(f"Test {idx + 1}: {test_case}")
