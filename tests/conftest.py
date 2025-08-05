import yaml
import json
import pytest
import os

@pytest.fixture
def config(config_path = 'params.yaml'):
    with open(config_path) as l:
        config = yaml.safe_load(l)
    return config

@pytest.fixture
def get_schema(schema_path = 'schema_in.json'):
    with open(schema_path) as s:
        schema = json.load(s)
    return schema