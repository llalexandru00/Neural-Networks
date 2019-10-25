import numpy as np

input_file = "in.in"


# function used to throw error message and stop execution
def fail(message):
    print("ERROR: " + message)
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

# works only with well formatted input
a = np.array(coefficients_matrix)
b = np.array(results_vector)
det_value = np.linalg.det(a)
if det_value == 0:
    print("incompatible")
else:
    x = np.linalg.solve(a, b)
    print(x)
