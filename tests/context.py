import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.palimpzest as palimpzest

# read the env file
with open(".env") as f:
    for line in f:
        key, value = line.strip().split("=")
        os.environ[key] = value
