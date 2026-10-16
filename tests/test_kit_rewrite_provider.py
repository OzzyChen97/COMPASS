from pathlib import Path
import pytest
from tools.build_kit_official_validation_rewrites import read_api_configuration,parse_response_json


def test_only_api_fields_are_used_and_v1_is_added(tmp_path):
    path=tmp_path/'credentials.txt'
    path.write_text('overleaf git token:\nnot-the-api-token\n\napi key:\ntest-api-value\napi url:\nhttps://example.invalid/\n')
    key,url=read_api_configuration(path)
    assert key=='test-api-value'
    assert url=='https://example.invalid/v1'


def test_existing_v1_is_preserved(tmp_path):
    path=tmp_path/'credentials.txt';path.write_text('API KEY:\nfake\nAPI URL:\nhttps://example.invalid/v1/\n')
    assert read_api_configuration(path)[1]=='https://example.invalid/v1'


def test_missing_or_credential_bearing_url_is_rejected(tmp_path):
    path=tmp_path/'credentials.txt';path.write_text('overleaf git token:\nunrelated\n')
    with pytest.raises(ValueError):read_api_configuration(path)
    path.write_text('api key:\nfake\napi url:\nhttps://user:password@example.invalid/\n')
    with pytest.raises(ValueError):read_api_configuration(path)


def test_fenced_json_response():
    assert parse_response_json('```json\n{"results": [true]}\n```')=={'results':[True]}


def test_short_caption_request_exposes_existing_length_filter():
    import json
    from tools.build_kit_official_validation_rewrites import constrained_request
    value=json.loads(constrained_request('Somebody walks lazy'))
    assert value['inputs'][0]['caption']=='Somebody walks lazy'
    assert 'between 1 and 9 whitespace-separated words' in value['constraints']