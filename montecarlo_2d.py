import cupy as cp
import cupyx as cpx


def monte_carlo_diffraction_cupy_2D(
    N_total,
    final_image_real,
    final_image_imag,
    source_width_x,
    source_width_y,
    slit_radius,
    det_yx,
    z_prop,
    wavelength,
    batch_size=int(1e7),
):  # Batch size of rays to GPU
    y_det = det_yx.shape[1]
    x_det = det_yx.shape[2]

    # Pdf to normalise ray amplitudes at beginning
    # for plane wave this is a constant for all rays
    pdf = 1 / (source_width_x * source_width_y)

    # Initialize arrays to count the number of rays per pixel
    counts = cp.zeros((y_det, x_det), dtype=cp.int32)
    for start_idx in range(0, N_total, batch_size):
        end_idx = min(start_idx + batch_size, N_total)
        current_batch_size = end_idx - start_idx

        # Get random pixel samples of the source and the detector
        source_x = cp.random.uniform(
            -source_width_x / 2, source_width_x / 2, size=current_batch_size
        )
        source_y = cp.random.uniform(
            -source_width_y / 2, source_width_y / 2, size=current_batch_size
        )
        source_z = cp.zeros(current_batch_size)

        # Remove rays that did not go through the
        # aperture and form an integer mask with it
        U = (
            cp.sqrt((source_x**2 + source_y**2)) < slit_radius
        ).astype(cp.int32)

        rand_det_idx_x = cp.random.randint(x_det, size=current_batch_size)
        rand_det_idx_y = cp.random.randint(y_det, size=current_batch_size)

        # Find counts of rays on detector
        cp.add.at(counts, (rand_det_idx_y, rand_det_idx_x), 1)

        # Organise coordinats on detector that rays hit
        # z y x is the ordering of coordinates
        det_x = det_yx[1, rand_det_idx_y, rand_det_idx_x]
        det_y = det_yx[0, rand_det_idx_y, rand_det_idx_x]
        det_z = cp.ones(current_batch_size) * z_prop

        # Convert coordinates to a vector for computation
        ray_source_coord = cp.array([source_z, source_y, source_x])
        ray_det_coord = cp.array([det_z, det_y, det_x])

        # Create ray vector and get magnitude and direction
        ray = ray_det_coord - ray_source_coord
        ray_distance = cp.linalg.norm(ray, axis=0)

        # Wavenumber
        k = 2 * cp.pi / wavelength

        # Get amplitude and phase of all rays
        amplitude = (
            U
            * (1 / ray_distance)
            * (1 / (2 * cp.pi))
            * (1j * k * z_prop / ray_distance)
            / pdf
        )
        phase = k * ray_distance

        # Get diffraction field at detector by summing
        # the point sources from the aperture.
        complex_rays = amplitude * (cp.exp(1j * phase))

        # Add complex wavefront to each pixel in the image
        cpx.scatter_add(
            final_image_real,
            (rand_det_idx_y, rand_det_idx_x),
            complex_rays.real,
        )
        cpx.scatter_add(
            final_image_imag,
            (rand_det_idx_y, rand_det_idx_x),
            complex_rays.imag,
        )

    # Add ampliude and phase together
    final_image = final_image_real + 1j * final_image_imag

    # Divide each pixel that a ray has hit by the number of counts
    non_zero_counts = counts > 0
    final_image[non_zero_counts] *= 1 / counts[non_zero_counts]

    # Include final factors which scale image by pixel size, and
    # have an extra term from the rayleigh sommerfeld integral.
    return final_image, ray_source_coord, ray_det_coord, U
