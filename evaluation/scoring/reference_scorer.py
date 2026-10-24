# Simple reference-based scoring
# responsible for assigning the binary labels 


import re


def contains_reference(response, scoring_target):
    """Return 1 if any scoring phrase appears, otherwise 0."""
    if not isinstance(response, str):
        raise TypeError("response must be a string")

    if isinstance(scoring_target, str):
        scoring_target = [scoring_target]

    for phrase in scoring_target:
        if not isinstance(phrase, str):
            raise TypeError("Each scoring target must be a string")

        phrase = phrase.strip()
        if not phrase:
            continue

        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        if re.search(pattern, response, flags=re.IGNORECASE):
            return 1

    return 0
