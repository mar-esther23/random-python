from math import factorial


def index_to_permutation(index, elements):
    """
    Returns the i-th permutation of elements.

    Arguments:
        elements    elements to permute
        index       index of the permutation

    Returns:
        res:        permuted list
    """

    pool = list(elements)
    res = list()
    while len(pool) > 0:
        n = len(pool)
        res.append( pool.pop((index) / factorial(n-1)) )
        index = index % (factorial(n-1))
    return res
