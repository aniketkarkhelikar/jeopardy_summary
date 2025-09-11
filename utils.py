def split_list(lst, n):
    """Splits a list into n roughly equal parts."""
    if n <= 0:
        raise ValueError("Number of parts must be positive.")
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]