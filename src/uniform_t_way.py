import itertools
import random
from z3 import *

def t_wise_testing(params, values, constraints, t=2):
    """
    Generate a t-wise test suite using Z3 with constraint handling.

    Args:
        params (list): List of parameter names.
        values (dict): Dictionary mapping each parameter to a list of possible values.
        constraints (list): List of constraint functions taking Z3 variables.
        t (int): The strength of interaction (e.g., 2 for pairwise, 3 for 3-wise).

    Returns:
        list: A list of test cases (dicts mapping parameter names to values).
    """

    def create_base_solver():
        z3_vars = {p: Int(p) for p in params}
        s = Solver()
        for p in params:
            s.add(z3_vars[p] >= 0, z3_vars[p] < len(values[p]))
        for c in constraints:
            s.add(c(z3_vars))
        return s, z3_vars

    # Generate all t-wise parameter combinations
    param_combos = list(itertools.combinations(params, t))

    # Track uncovered value combinations for each t-wise parameter combination
    uncovered = {}
    for combo in param_combos:
        value_combos = list(itertools.product(*(values[p] for p in combo)))
        uncovered[combo] = set(value_combos)

    pairs = []
    test_suite = []

    while any(uncovered[combo] for combo in param_combos):
        # Find the parameter combo with the most uncovered value combinations
        best_combo = max(uncovered.items(), key=lambda x: len(x[1]) if x[1] else -1)[0]
        value_combos = list(uncovered[best_combo])
        random.shuffle(value_combos)

        found = False
        for val_combo in value_combos:
            s, z3_vars = create_base_solver()
            for p, v in zip(best_combo, val_combo):
                s.add(z3_vars[p] == values[p].index(v))
            if s.check() == sat:
                found = True
                break
            else:
                uncovered[best_combo].remove(val_combo)

        if not found:
            continue


        # Build partial test case from the chosen t-wise combination
        test_case = {p: v for p, v in zip(best_combo, val_combo)}
        assigned = set(test_case.keys())

        s, z3_vars = create_base_solver()
        for p, v in test_case.items():
            s.add(z3_vars[p] == values[p].index(v))

        # Assign remaining parameters
        for p in params:
            if p in assigned:
                continue
            candidates = []
            for v in values[p]:
                s.push()
                s.add(z3_vars[p] == values[p].index(v))
                if s.check() == sat:
                    # Score based on how many new t-wise combinations this value would cover
                    score = 0
                    for combo in param_combos:
                        if p not in combo:
                            continue
                        if all(param in test_case for param in combo if param != p):
                            full_combo = tuple(
                                test_case[param] if param != p else v for param in combo
                            )
                            if full_combo in uncovered[combo]:
                                score += 1
                    candidates.append((v, score))
                s.pop()
            if not candidates:
                raise Exception(f"No valid value for parameter {p} under constraints.")
            max_score = max(c[1] for c in candidates)
            best_vals = [v for v, s in candidates if s == max_score]
            chosen_val = random.choice(best_vals)
            test_case[p] = chosen_val
            s.add(z3_vars[p] == values[p].index(chosen_val))


        # Add the test case to the suite if it's not already present
        sorted_test_case = {k: test_case[k] for k in sorted(test_case.keys())}
        if sorted_test_case not in [dict(sorted(tc.items())) for tc in test_suite]:
            test_suite.append(sorted_test_case)
            
        # Update uncovered combinations
        for combo in param_combos:
            if all(p in test_case for p in combo):
                val_combo = tuple(test_case[p] for p in combo)
                uncovered[combo].discard(val_combo)

    return test_suite

