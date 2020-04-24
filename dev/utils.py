import os


def relpath(path: str) -> str:
    return os.path.join(os.path.dirname(__file__), path)
