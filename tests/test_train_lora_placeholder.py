import os
import requests


def test_train_lora_placeholder():
    os.environ['OELALA_FORCE_LORA_PLACEHOLDER'] = '1'
    url = 'http://127.0.0.1:7999/train-lora-placeholder'
    import subprocess, shlex, json
    cmd = (
        "curl -s -X POST 'http://127.0.0.1:7999/train-lora-placeholder' "
        "-F 'files=@/home/flip/oelala/uploads/dummy1.png' "
        "-F 'files=@/home/flip/oelala/uploads/dummy2.png' "
        "-F 'model_name=pytest_lora'"
    )
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert proc.returncode == 0
    j = json.loads(proc.stdout)
    assert j.get('success') is True
    assert 'lora_path' in j
    print('placeholder test passed, lora_path=', j.get('lora_path'))
