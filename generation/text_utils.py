from __future__ import annotations


def underscore_to_pascalcase(underscore_str: str) -> str:
    if not underscore_str:
        return ""
    parts = underscore_str.split("_")
    return "".join(word.capitalize() for word in parts if word)


def read_file(file_path: str) -> str:
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        print(f"Error reading file {file_path}: {e}")
        return ""
