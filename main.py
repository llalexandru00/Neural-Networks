input_file = "in.in"
epsilon = 1e-9


# function used to throw error message and stop execution
def fail(message):
    print("ERROR: " + message)
    exit()


# function used to throw error message and stop execution
def success(message, output_solution=()):
    print("RESULT: " + message)
    if len(output_solution) > 0:
        print("Example solution: ")
        print(output_solution)
    exit()


# responsible for parsing one single term; ideally it should have a starting number (optional) and a variable name
# both will be returned in a form of a map
def parse(term):
    coefficient = 0
    var_name = ""
    parsing_coefficient = True
    empty_coefficient = True
    negative = False

    if term == "":
        fail("There are empty terms. Make sure there are terms between each binary operator (+/-) "
             "and the left side of the equation is not empty")

    if term[0] == '-':
        negative = True
        term = term[1:]

    for i in term:
        if parsing_coefficient and i.isnumeric():
            empty_coefficient = False
            coefficient = coefficient * 10 + (ord(i) - ord('0'))
        else:
            parsing_coefficient = False
            if i.isalnum():
                var_name = var_name + i
            else:
                fail("The variable name has invalid characters: " + var_name + i)

    # special case in which the coefficient is not mentioned
    if empty_coefficient:
        coefficient = 1

    return {"coefficient": coefficient * (-1 if negative else 1), "var_name": var_name}


# responsible for returning a list of terms in form of a tuple with the coefficient and variable name
def get_terms(expression):
    terms = expression.split()
    parsed_terms = []

    if len(terms) % 2 == 0:
        fail("A line is not well formatted. There should be spaces only around the binary operators (+/-).")

    for i in range(0, len(terms), 2):
        term = parse(terms[i])
        parsed_terms.append(term)

    for i in range(1, len(terms), 2):
        if terms[i] == '+':
            continue
        elif terms[i] == '-':
            parsed_terms[(i + 1) // 2]["coefficient"] *= -1
        else:
            fail("The expression in the left side is invalid. Between two terms there must be " +
                 "exactly one binary operator (+/-).")

    return parsed_terms


# parses the result expression to return an integer
def get_result(expression):
    value = 0
    negative = False

    if expression == "":
        fail("The result is non-existent. Make sure there is a valid number in the right side of the equation")

    if expression[0] == '-':
        negative = True
        expression = expression[1:]

    for i in expression:
        if i.isnumeric():
            value = value * 10 + (ord(i) - ord('0'))
        else:
            fail("The result of one equation is not a valid number")

    return value * (-1 if negative else 1)


# parses the matrix and remove free terms if any by subtracting them from the result
def process_free_terms(lines, results):
    processed_matrix = []
    processed_results = []

    for i in range(0, len(lines)):
        aux_value = 0
        line = lines[i]
        processed_line = []

        for term in line:
            if term["var_name"] == "":
                aux_value += term["coefficient"]
            else:
                processed_line.append(term)

        processed_matrix.append(processed_line)
        processed_results.append(results[i] - aux_value)

    return processed_matrix, processed_results


# converts a parsed line into a dictionary storing the variable name and its coefficient
def get_mapping(line):
    mapping = {}
    for term in line:
        mapping[term["var_name"]] = term["coefficient"]
    return mapping


# converts the parsed matrix into a coefficient matrix and returns it along with a variable list in order
def get_coefficients_matrix(lines):
    variables = set()

    for line in lines:
        for term in line:
            variables.add(term["var_name"])

    variables = sorted(list(variables))
    matrix = [[0 for _ in variables] for _ in lines]

    for i in range(0, len(lines)):
        line_mapping = get_mapping(lines[i])
        for j in range(0, len(variables)):
            if variables[j] in line_mapping.keys():
                matrix[i][j] = line_mapping[variables[j]]

    return matrix, variables


# attaches the result vector at the end of the matrix
def get_extended_matrix(matrix, results):
    extended_matrix = []

    for i in range(0, len(results)):
        line = matrix[i][:]
        line.append(results[i])
        extended_matrix.append(line)

    return extended_matrix


# swaps two rows inside the matrix
def swap_rows(matrix, row1, row2):
    row = matrix[row1]
    matrix[row1] = matrix[row2]
    matrix[row2] = row


# returns a hard copy of a matrix
def hard_copy(matrix):
    copy = []
    for line in matrix:
        copy.append(line[:])
    return copy


# check if a line has only null elements
def is_empty_line(line):
    for c in line:
        if c > epsilon or c < -epsilon:
            return False
    return True


# returns the reduced matrix and the rank of it in O(N*M*min(N, M))
def reduce_row_echelon_form(matrix):
    # hard copy the matrix
    nr_lines = len(matrix)
    nr_cols = len(matrix[0])
    current_line = 0
    current_column = 0
    rank = min(nr_lines, nr_cols)

    while current_line < rank and current_column < nr_cols:
        # find the first row which has a not null pivot
        if matrix[current_line][current_column] == 0:
            for k in range(current_line+1, nr_lines):
                if matrix[k][current_column] != 0:
                    swap_rows(matrix, current_line, k)
                    break

        pivot = matrix[current_line][current_column]
        if -epsilon < pivot < epsilon:
            # the pivot can end up being null
            empty = True
            # check if the row is empty
            for j in range(current_line+1, nr_cols):
                if matrix[current_line][j] > epsilon or matrix[current_line][j] < -epsilon:
                    empty = False
                    break
            if empty and current_line < nr_cols - 1:
                swap_rows(matrix, current_line, rank-1)
                rank -= 1
        else:
            # apply gaussian elimination
            for k in range(current_line+1, nr_lines):
                multiplier = matrix[k][current_column] / pivot
                for j in range(current_column, nr_cols):
                    matrix[k][j] -= multiplier * matrix[current_line][j]
            current_line = current_line + 1
        current_column = current_column + 1

    while rank > 0 and is_empty_line(matrix[rank-1]):
        rank -= 1

    return rank


# checks if the system is compatible
def compatible(matrix, results):
    matrix = hard_copy(matrix)
    extended_matrix = get_extended_matrix(matrix, results)
    mat_rank = reduce_row_echelon_form(matrix)
    extended_rank = reduce_row_echelon_form(extended_matrix)
    return mat_rank == extended_rank


# extracts the minor which has the determinant different from 0 and the variables associated
def get_valid_minor(matrix, vars_vector):
    col_indexes = []
    for line in matrix:
        for i in range(0, len(line)):
            if line[i] > epsilon or line[i] < -epsilon:
                col_indexes.append(i)
                break

    result_minor = []
    for line in matrix:
        minor_line = []
        for i in col_indexes:
            minor_line.append(line[i])
        result_minor.append(minor_line)

    variables = []
    for i in col_indexes:
        variables.append(vars_vector[i])

    return result_minor, variables


# returns a minor obtained by removing from the matrix a line and a column both specified
def get_minor(matrix, line, col):
    computed_matrix = []
    for i in range(0, len(matrix)):
        if i == line:
            continue
        computed_line = []
        for j in range(0, len(matrix[i])):
            if j == col:
                continue
            computed_line.append(matrix[i][j])
        computed_matrix.append(computed_line)
    return computed_matrix


# obtain determinant out of the reduced row echelon form, by multiplying the elements on the second diagonal
def get_determinant(matrix):
    rank = reduce_row_echelon_form(matrix)
    if rank == 0:
        return 0
    det_value = 1
    for i in range(0, len(matrix)):
        det_value *= matrix[i][i]
    return det_value


# get the adj matrix
def get_adj_matrix(matrix):
    adj_matrix = [[0 for _ in line] for line in matrix]
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            related_minor = get_minor(matrix, i, j)
            adj_matrix[i][j] = get_determinant(related_minor) * (1 if (i+j) % 2 == 0 else -1)
    return adj_matrix


# transpose the matrix
def transpose(matrix):
    for i in range(0, len(matrix)):
        for j in range(0, i):
            aux = matrix[i][j]
            matrix[i][j] = matrix[j][i]
            matrix[j][i] = aux


# multiply a matrix to a vector
def multiply(matrix, vec):
    result = [0 for _ in vec]
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            result[i] += matrix[i][j] * vec[j]
    return result


# obtain the solution in form of a map which maps the variable name to the value
def compute_solution(matrix, var_vec, results):
    # the minor given should be in a reduced square row echelon form
    det_value = 1
    for i in range(0, len(matrix)):
        det_value *= matrix[i][i]
    assert det_value < -epsilon or epsilon < det_value

    adj_matrix = get_adj_matrix(matrix)
    transpose(adj_matrix)
    for i in range(0, len(adj_matrix)):
        for j in range(0, len(adj_matrix[i])):
            adj_matrix[i][j] *= 1/det_value

    sol_vec = multiply(adj_matrix, results)
    sol = []
    for i in range(0, len(sol_vec)):
        sol.append({var_vec[i]: sol_vec[i]})
    return sol


content = open(input_file)
input_lines = [line.strip() for line in content.readlines()]

terms_collection = []
results_vector = []
for input_line in input_lines:
    expressions = input_line.split("=")
    if len(expressions) != 2:
        fail("A line should have exactly one equal sign (=).")
    terms_collection.append(get_terms(expressions[0].strip()))
    results_vector.append(get_result(expressions[1].strip()))

terms_collection, results_vector = process_free_terms(terms_collection, results_vector)
coefficients_matrix, var_vector = get_coefficients_matrix(terms_collection)

if not compatible(coefficients_matrix, results_vector):
    success("The system is incompatible.")

matrix_rank = reduce_row_echelon_form(coefficients_matrix)
coefficients_matrix = [line for line in coefficients_matrix[:matrix_rank]]
results_vector = [i for i in results_vector[:matrix_rank]]

if matrix_rank == 0:
    success("The system is indeterminate.", [{var_name: 0} for var_name in var_vector])

minor, minor_var_vector = get_valid_minor(coefficients_matrix, var_vector)
solution = compute_solution(minor, minor_var_vector, results_vector)

if len(coefficients_matrix) < len(coefficients_matrix[0]):
    determined_vars = set(minor_var_vector)
    for var in var_vector:
        if var not in determined_vars:
            solution.append({var: 0})
    success("The system is indeterminate.", solution)
else:
    success("The system is determinate.", solution)
