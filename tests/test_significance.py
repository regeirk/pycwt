import numpy
import pytest

from pycwt import significance


@pytest.mark.parametrize(
    "dof",
    [
        128,
        128.0,
        numpy.int64(128),
        numpy.float64(128),
        numpy.array(128),
        numpy.array([128]),
        [128],
        (128,),
    ],
)
def test_time_averaged_scalar_dof(dof):
    before = numpy.array(dof, copy=True)
    actual, _ = significance(
        1.0,
        1.0,
        numpy.array([2.0, 4.0, 8.0, 16.0]),
        sigma_test=1,
        alpha=0.0,
        dof=dof,
    )
    # Equation23 with Morlet(6). Quantiles independently computed by
    # 60-digit regularized incomplete-gamma CDF inversion.
    expected = [
        1.332272815014519,
        1.479459270452457,
        1.693455335290468,
        1.995594566246947,
    ]
    numpy.testing.assert_allclose(actual, expected, rtol=1e-12)
    numpy.testing.assert_array_equal(dof, before)


def test_time_averaged_default_dof():
    actual, _ = significance(
        1.0,
        1.0,
        numpy.array([2.0, 4.0, 8.0, 16.0]),
        sigma_test=1,
        alpha=0.0,
    )
    expected = [
        2.907234502192159,
        2.971816623436183,
        2.989627487312172,
        2.994197961241597,
    ]
    numpy.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_time_averaged_per_scale_dof():
    dof = numpy.full(4, 128)
    actual, _ = significance(
        1.0,
        1.0,
        numpy.array([2.0, 4.0, 8.0, 16.0]),
        sigma_test=1,
        alpha=0.0,
        dof=dof,
    )
    expected = [
        1.332272815014519,
        1.479459270452457,
        1.693455335290468,
        1.995594566246947,
    ]
    numpy.testing.assert_allclose(actual, expected, rtol=1e-12)
    numpy.testing.assert_array_equal(dof, numpy.full(4, 128))


@pytest.mark.parametrize(
    "dof, dt, expected",
    [
        (
            numpy.uint8(128),
            2,
            [
                1.231241415974839,
                1.332272815014519,
                1.479459270452457,
                1.693455335290468,
            ],
        ),
        (
            numpy.float32(128),
            0.1,
            [
                2.112021968509944,
                2.497498550264070,
                2.798701426791266,
                2.936997819687546,
            ],
        ),
    ],
)
def test_time_averaged_small_dtype_dof(dof, dt, expected):
    actual, _ = significance(
        1.0,
        dt,
        numpy.array([2.0, 4.0, 8.0, 16.0]),
        sigma_test=1,
        alpha=0.0,
        dof=dof,
    )
    numpy.testing.assert_allclose(actual, expected, rtol=1e-12)
