import numpy as np
import cupy as cp
from montecarlo_2d import monte_carlo_diffraction_cupy_2D


def run_notebook():
    um = 1e-6

    # Set up coordinates
    aperture_diameter = 100 * um
    detector_width_x = 500 * um
    detector_width_y = 500 * um

    # Define number of pixels on detector
    num_px_det_x = 125
    num_px_det_y = 125

    # Set detector width
    detector_px_width_x = detector_width_x / num_px_det_x
    detector_px_width_y = detector_width_y / num_px_det_y

    num_pixels = int(num_px_det_x * num_px_det_y)
    det_yx_indices = np.indices(
        (num_px_det_y, num_px_det_x)
    ).reshape(2, num_pixels).T

    # Formd the detector coordinates matrix
    det_yx_coords = np.zeros(det_yx_indices.shape, dtype=np.float64)
    det_yx_coords[:, 0] = (
        det_yx_indices[:, 0] * detector_px_width_y
        - detector_width_y / 2
        + detector_px_width_y / 2
    )
    det_yx_coords[:, 1] = (
        det_yx_indices[:, 1] * detector_px_width_x
        - detector_width_x / 2
        + detector_px_width_x / 2
    )
    det_yx_coords = det_yx_coords.T.reshape(2, num_px_det_y, num_px_det_x)

    # Define propagation distance after aperture
    z_prop = 500 * um

    # Set wavelength
    wavelength = 1 * um

    # Choose number of rays
    num_rays = int(1e10)

    # Set source area
    source_width_x = detector_width_x
    source_width_y = detector_width_y

    # Make cupy arrays
    det_yx_coords = cp.asarray(det_yx_coords)

    # Initialize final image and aperture phase shifts arrays on GPU
    final_image_real = cp.zeros([num_px_det_y, num_px_det_x], dtype=cp.float64)
    final_image_imag = cp.zeros([num_px_det_y, num_px_det_x], dtype=cp.float64)

    final_image, ray_aper_coord, ray_det_coord, aperture_mask = (
        monte_carlo_diffraction_cupy_2D(
            num_rays,
            final_image_real,
            final_image_imag,
            source_width_x,
            source_width_y,
            slit_radius=aperture_diameter / 2,
            det_yx=det_yx_coords,
            z_prop=z_prop,
            wavelength=wavelength,
            batch_size=int(1e7),
        )
    )
    # Convery back to numpy for plotting
    det_yx_coords = cp.asnumpy(det_yx_coords)
    final_image = cp.asnumpy(final_image)
    ray_aper_coord = cp.asnumpy(ray_aper_coord)
    ray_det_coord = cp.asnumpy(ray_det_coord)
    aperture_mask = cp.asnumpy(aperture_mask)


def test_notebook(benchmark):
    benchmark(run_notebook)


if __name__ == "__main__":
    run_notebook()
