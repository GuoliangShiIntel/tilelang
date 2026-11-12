import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "e2e"))
from decode import DeviceModel


def test_decode_ba_dpu_rejects_per_layer_mode():
    from decode import main

    with pytest.raises(SystemExit) as error:
        main(["--decode-gdn-ba-dpu", "--per-layer"])
    assert error.value.code == 2


@pytest.mark.parametrize("count", [1, 4, 5, 9])
def test_standalone_head_reads_last_live_row_without_decode(count):
    model = DeviceModel.__new__(DeviceModel)
    model.c = SimpleNamespace(hidden=8)
    model.ps = {"NP": 4}
    model.pos = 0
    calls = []
    hidden = np.arange(8, dtype=np.float32).reshape(1, 8)

    def prefill(ids):
        assert ids == list(range(count))
        model.pos += len(ids)
        return 12.0

    def download(name, dtype, shape, offset=0):
        if name == "P_Hio":
            assert offset == ((count - 1) % 4) * 8 * 4
            assert shape == (1, 8)
            return hidden
        assert name == "H_Tok"
        return np.full(shape, 17, dtype=dtype)

    def upload(name, value):
        assert name == "Hio"
        np.testing.assert_array_equal(value, hidden)

    model.prefill = prefill
    model._bufs = lambda kind, layer: ["Hio"]
    model.s = SimpleNamespace(download=download, upload=upload, launch=lambda *args: calls.append(args) or (2.0, 1.0))
    token, timing, pre_ms = model.prefill_prompt(list(range(count)), True)
    assert token == 17
    assert timing["head"] == 1.0
    assert pre_ms == 12.0
    assert model.pos == count
    assert calls == [("head", "Hio")]


def test_standalone_head_rejects_empty_prompt():
    with pytest.raises(ValueError, match="at least one"):
        DeviceModel.__new__(DeviceModel).prefill_prompt([], True)


@pytest.mark.parametrize("ids", [[3], [3, 5, 7]])
def test_legacy_prefill_keeps_last_token_for_decode(ids):
    model = DeviceModel.__new__(DeviceModel)
    calls = []
    model.prefill = lambda tokens: calls.append(tokens) or 12.0
    model.step = lambda token: calls.append(token) or (17, {"model": 4.0})
    token, timing, pre_ms = model.prefill_prompt(ids)
    assert token == 17
    assert timing == {"model": 4.0}
    assert pre_ms == (12.0 if len(ids) > 1 else 0.0)
    assert calls == ([ids[:-1], ids[-1]] if len(ids) > 1 else [ids[-1]])
