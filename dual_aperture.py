from typing import NamedTuple
import numpy as np  # noqa
import cupy as cp  # noqa
import tqdm.auto as tqdm
from dataclasses import dataclass

xp = cp

if xp == cp:
    from cupyx.scipy.interpolate import RegularGridInterpolator
else:
    from scipy.interpolate import RegularGridInterpolator


class Aperture(NamedTuple):
    z: float
    radius: float
    cyx: tuple[float, float] | None

    def random_on(self, num: int, rng: xp.random.RandomState):
        random = rng.random_sample(
            size=(num, 2)
        )
        random *= xp.asarray((
            self.radius ** 2,
            2 * xp.pi,
        ), dtype=xp.float32)
        source_r = xp.sqrt(random[:, 0])
        source_y = source_r * xp.sin(random[:, 1])
        source_x = source_r * xp.cos(random[:, 1])
        if self.cyx is not None:
            source_y += self.cyx[0]
            source_x += self.cyx[1]
        return xp.stack(
            (source_y, source_x),
            axis=1,
        )

    def _phase_shift(self, wave, pos_yx):
        return


@dataclass
class Sample:
    z: float
    size: tuple[float, float]
    illuminated_r: float
    illuminated_cyx: tuple[float, float] | None
    phase_shift: xp.ndarray  # premultiplied by xp.exp(1j)!
    periodic: bool

    def _get_interpolator(self):
        try:
            return self._interpolator
        except AttributeError:
            py, px = self.phase_shift.shape
            sy, sx = xp.asarray(self.size) / 2.
            yv = xp.linspace(-sy, sy, num=py, endpoint=True)
            xv = xp.linspace(-sx, sx, num=px, endpoint=True)
            self._interpolator = RegularGridInterpolator(
                (yv, xv),
                self.phase_shift,
                method='linear',
                bounds_error=False,
                fill_value=0.,
            )
            return self._interpolator

    # def random_on(self, num: int, rng: xp.random.RandomState):
    #     random = rng.random_sample(
    #         size=(num, 2)
    #     )
    #     random -= 0.5
    #     random *= (
    #         2
    #         * xp.asarray(self.size, dtype=xp.float32)
    #     )
    #     return random

    def random_on(self, num: int, rng: xp.random.RandomState):
        random = rng.random_sample(
            size=(num, 2)
        )
        random *= xp.asarray((
            self.illuminated_r ** 2,
            2 * xp.pi,
        ), dtype=xp.float32)
        source_r = xp.sqrt(random[:, 0])
        source_y = source_r * xp.sin(random[:, 1])
        source_x = source_r * xp.cos(random[:, 1])
        if self.illuminated_cyx is not None:
            source_y += self.illuminated_cyx[0]
            source_x += self.illuminated_cyx[1]
        return xp.stack(
            (source_y, source_x),
            axis=1,
        )

    def _phase_shift(self, wave, pos_yx):
        interpolator = self._get_interpolator()
        if self.periodic:
            size = xp.asarray(self.size)[np.newaxis, :]
            pos_yx = pos_yx + size / 2.
            pos_yx %= size
            pos_yx -= size / 2.
        wave *= interpolator(pos_yx)


@dataclass
class Detector:
    z: float
    shape: tuple[int, int]
    size: tuple[float, float]

    @property
    def num_pix(self):
        return xp.prod(xp.asarray(self.shape))

    def get_array(self) -> xp.ndarray:
        return xp.zeros(self.shape, dtype=xp.complex128)

    def build_coords_yx(self):
        py, px = self.shape
        sy, sx = xp.asarray(self.size) / 2.
        yv, xv = xp.meshgrid(
            xp.linspace(-sy, sy, num=py, endpoint=True),
            xp.linspace(-sx, sx, num=px, endpoint=True),
        )
        return xp.stack((yv, xv), axis=-1).reshape(-1, 2)

    @property
    def coords_yx(self) -> xp.ndarray:
        try:
            return self._coords_yx
        except AttributeError:
            self._coords_yx = self.build_coords_yx()
            return self._coords_yx

    def accumulate_on(
        self,
        cimage,
        source_yx,
        source_z,
        source_area,
        num_rays,
        source_wave,
        wavelength,
        rng=None,
    ):
        source_yx = source_yx[xp.newaxis, ...]  # (1, batch_size, 2)
        pixel_yx = self.coords_yx[:, xp.newaxis, ...]  # (num_pixels, 1, 2)
        z_prop = abs(source_z - self.z)
        delta_wave, _ = propagate(
            pixel_yx - source_yx,
            z_prop,
            wavelength,
        )  # (num_pixels, batch_size)
        delta_wave *= (source_area / num_rays)
        all_waves = (
            source_wave[xp.newaxis, ...]
            * delta_wave
        )
        cimage += all_waves.sum(axis=-1).reshape(self.shape)


class PlaneWaveDetector(Detector):
    def accumulate_on(
        self,
        accumulator,
        source_yx,
        source_z,
        source_wave,
        wavelength,
        rng=None,
    ):
        if rng is None:
            rng = xp.random.RandomState()
        batch_size = source_yx.shape[0]
        pixel_idcs = rng.randint(
            0,
            self.num_pix,
            size=(batch_size,),
        )
        all_pixels_yx = self.coords_yx
        dest_yx = all_pixels_yx[pixel_idcs]
        z_prop = abs(source_z - self.z)
        dyx = dest_yx - source_yx  # (batch_size, 2)
        delta_wave, ray_distance = propagate(
            dyx,
            z_prop,
            wavelength,
        )
        # vector to all other pixels from every pixel hit by ray, in-plane
        # Since the detector is parallel to the axis the z-component would be
        # zero so we can avoid computing it
        ray_px_to_pixels = (
            all_pixels_yx[:, np.newaxis, :]
            - dest_yx[np.newaxis, ...]
        )  # (num_pixels, batch_size, 2)
        # we only need the y/x components of the unit vector
        # to do the dot product with ray_px_to_pixels (batch_size, 2)
        ray_unit_vector = dyx / ray_distance[:, np.newaxis]
        # implements dot product over the axis needed
        dist_in_ray_direction = xp.sum(
            ray_px_to_pixels * ray_unit_vector[np.newaxis, ...],
            axis=-1,
        )  # (num_pixels, batch_size)
        phase_shift = 2 * xp.pi * dist_in_ray_direction / wavelength
        # (num_pixels, batch_size)
        delta_wave_pixels = (
            delta_wave[np.newaxis, ...]
            * xp.exp(1j * phase_shift)
        )
        # (num_pixels, batch_size)
        all_waves = source_wave[np.newaxis, ...] + delta_wave_pixels
        accumulator += all_waves.sum(axis=-1).reshape(self.shape)


def propagate(dyx, z_prop, wavelength):
    k = 2 * xp.pi / wavelength
    ray_distance2 = xp.sum(
        dyx ** 2,
        axis=-1,
    )

    ray_distance2 += z_prop ** 2
    ray_distance = xp.sqrt(
        ray_distance2,
    )

    # Get the change to the complex ray over this propagation
    wave = (
        (-1j + 1 / (k * ray_distance))
        * z_prop
        * xp.exp(
            1j * k * ray_distance,
        )
        / (wavelength * ray_distance2)
    )
    return wave, ray_distance


def run_model(
    num_rays: int,
    wavelength: float,
    a0: Aperture,
    a1: Aperture | None,
    detector: Detector,
    batch_size=int(1e2),
) -> xp.ndarray:

    rng = xp.random.RandomState()
    image = detector.get_array()

    batch_size = min(batch_size, num_rays)
    for _ in tqdm.trange(max(1, num_rays // batch_size)):
        wave = xp.ones(
            (batch_size,),
            dtype=xp.complex128,
        )
        source = a0.random_on(batch_size, rng)
        source_z = a0.z
        # a0._phase_shift(wave, source)

        if a1 is not None:
            dest = a1.random_on(batch_size, rng)
            a1._phase_shift(wave, dest)
            delta_wave, _ = propagate(
                dest - source,
                abs(source_z - a1.z),
                wavelength
            )
            wave *= xp.exp(1j * xp.angle(delta_wave))
            source = dest
            source_z = a1.z

        source_area = xp.pi * a0.radius ** 2
        detector.accumulate_on(
            image,
            source,
            source_z,
            source_area,
            num_rays,
            wave,
            wavelength,
            rng=rng,
        )

    try:
        image = image.get()
    except AttributeError:
        pass

    return image


if __name__ == '__main__':
    um = 1e-6

    wavefront = run_model(
        int(1e3),
        1 * um,
        Aperture(
            z=0 * um, radius=100 * um, cyx=None,
        ),
        Aperture(
            z=250 * um, radius=100 * um, cyx=(10 * um, 10 * um)
        ),
        Detector(
            z=500 * um, shape=(256, 256), size=(500 * um, 500 * um)
        ),
    )
