"""Request-level retention is limited to forced runs on the measured target."""
from types import SimpleNamespace

import pytest

from onpu.speculation import configure_request_retention


def measured_decoder():
    draft = SimpleNamespace(observe_features=lambda *_args, **_kwargs: None)
    return SimpleNamespace(serving_setup={"target_table": {"shared_table": True}},
                           retain_fallback_prefix=False, fallback_observation_batch_size=1,
                           draft=draft)


def test_force_enables_batched_retention_for_measured_shared_table():
    decoder = measured_decoder()
    configure_request_retention(decoder, "force")
    assert decoder.retain_fallback_prefix is True
    assert decoder.fallback_observation_batch_size == 8


@pytest.mark.parametrize("mode", ["auto", "off"])
def test_nonforced_request_clears_retention_after_prior_force(mode):
    decoder = measured_decoder()
    configure_request_retention(decoder, "force")
    configure_request_retention(decoder, mode)
    assert decoder.retain_fallback_prefix is False
    assert decoder.fallback_observation_batch_size == 1


@pytest.mark.parametrize("decoder", [
    None,
    SimpleNamespace(serving_setup=object(), retain_fallback_prefix=False, fallback_observation_batch_size=1),
    SimpleNamespace(serving_setup={"target_table": object()}, retain_fallback_prefix=False, fallback_observation_batch_size=1),
    SimpleNamespace(serving_setup={"target_table": {"shared_table": False}},
                    retain_fallback_prefix=False, fallback_observation_batch_size=1,
                    draft=SimpleNamespace(observe_features=lambda: None)),
    SimpleNamespace(serving_setup={"target_table": {"shared_table": True}},
                    retain_fallback_prefix=0, fallback_observation_batch_size=1,
                    draft=SimpleNamespace(observe_features=lambda: None)),
    SimpleNamespace(serving_setup={"target_table": {"shared_table": True}},
                    retain_fallback_prefix=False, fallback_observation_batch_size=True,
                    draft=SimpleNamespace(observe_features=lambda: None)),
    SimpleNamespace(serving_setup={"target_table": {"shared_table": True}},
                    retain_fallback_prefix=False, fallback_observation_batch_size=1,
                    draft=SimpleNamespace()),
])
def test_unsupported_or_unmeasured_decoder_is_unchanged(decoder):
    if decoder is not None:
        before = (decoder.retain_fallback_prefix, decoder.fallback_observation_batch_size)
    configure_request_retention(decoder, "force")
    if decoder is not None:
        assert (decoder.retain_fallback_prefix, decoder.fallback_observation_batch_size) == before


def test_configuration_does_not_import_device_libraries(monkeypatch):
    import builtins

    original = builtins.__import__
    def without_devices(name, *args, **kwargs):
        if name.split(".")[0] in ("tinygrad", "torch"):
            raise AssertionError("Request configuration imported a device library")
        return original(name, *args, **kwargs)
    decoder = measured_decoder()
    with monkeypatch.context() as local:
        local.setattr(builtins, "__import__", without_devices)
        configure_request_retention(decoder, "force")
        configure_request_retention(decoder, "auto")
    assert decoder.retain_fallback_prefix is False
    assert decoder.fallback_observation_batch_size == 1
