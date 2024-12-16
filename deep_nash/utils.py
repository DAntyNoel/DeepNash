def batch_info_dicts(info_dicts):
    """
    Batch a list of info dictionaries into a single dictionary with lists of values.

    Args:
        info_dicts (list[dict]): A list of dictionaries with the same structure.

    Returns:
        dict: A single dictionary where each key maps to a list of corresponding values from the input dictionaries.
    """
    if not info_dicts:
        return {}

    # Ensure all dictionaries have the same keys
    keys = info_dicts[0].keys()
    if not all(d.keys() == keys for d in info_dicts):
        raise ValueError("All dictionaries in info_dicts must have the same keys.")

    return {key: [info[key] for info in info_dicts] for key in keys}