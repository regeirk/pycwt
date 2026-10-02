import numpy
import pytest

from pycwt import wavelet


@pytest.fixture
def significance_request(tmp_path, monkeypatch):
    monkeypatch.setattr(wavelet, "get_cache_dir", lambda: str(tmp_path))
    request = dict(
        al1=0.5,
        al2=0.5,
        dt=1,
        dj=0.25,
        s0=2,
        J=4,
        wavelet=wavelet.Morlet(6),
        mc_count=3,
        progress=False,
    )
    state = numpy.random.get_state()
    try:
        yield request
    finally:
        numpy.random.set_state(state)


@pytest.mark.parametrize(
    "changed",
    [
        dict(al1=0.99, al2=0.99),
        dict(al1=-0.5),
        dict(significance_level=0.99),
        dict(significance_level=numpy.nextafter(0.95, 1)),
        dict(mc_count=4),
    ],
)
def test_changed_request_does_not_reuse_cache(
    significance_request, monkeypatch, changed
):
    numpy.random.seed(27)
    wavelet.wct_significance(**significance_request)
    request = dict(significance_request, **changed)

    calls = []
    rednoise = wavelet.rednoise

    def record_rednoise(*args):
        calls.append(args)
        return rednoise(*args)

    monkeypatch.setattr(wavelet, "rednoise", record_rednoise)
    numpy.random.seed(42)
    cached = wavelet.wct_significance(**request)
    assert len(calls) == 1 + 2 * request["mc_count"]
    numpy.random.seed(42)
    uncached = wavelet.wct_significance(**request, cache=False)
    numpy.testing.assert_allclose(cached, uncached, equal_nan=True)


@pytest.mark.parametrize("coefficient", [-0.99, -0.25, 0.25, 0.99])
def test_cache_coefficient_transform_is_finite(significance_request, coefficient):
    request = dict(significance_request, al1=coefficient, al2=coefficient)
    with numpy.errstate(invalid="raise", divide="raise"):
        wavelet.wct_significance(**request)


def test_identical_request_reuses_cache(significance_request, monkeypatch):
    first = wavelet.wct_significance(**significance_request)

    def unexpected_simulation(*args):
        pytest.fail("A cached request should not generate new noise")

    monkeypatch.setattr(wavelet, "rednoise", unexpected_simulation)
    second = wavelet.wct_significance(**significance_request)
    numpy.testing.assert_array_equal(first, second)


def test_cache_false_does_not_access_cache(significance_request, monkeypatch):
    def unexpected_cache_access():
        pytest.fail("An uncached request should not access the cache directory")

    monkeypatch.setattr(wavelet, "get_cache_dir", unexpected_cache_access)
    wavelet.wct_significance(**significance_request, cache=False)


def test_legacy_cache_is_not_reused(significance_request, tmp_path):
    legacy = tmp_path / "wct_sig_nan_nan_0.25000_2.00000_4_Morlet.gz"
    numpy.savetxt(legacy, numpy.full(5, -1.0))
    result = wavelet.wct_significance(**significance_request)
    assert not numpy.array_equal(result, numpy.full(5, -1.0))
