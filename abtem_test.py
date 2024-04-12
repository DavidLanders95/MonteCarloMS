import abtem.multislice
import ase
import abtem
import matplotlib.pyplot as plt

import dask
dask.config.set(scheduler='synchronous')  # overwrite default with single-threaded scheduler

x1 = 1.0
atom_spacing = 3.0
x2 = x1 + atom_spacing
x_centre = 5.0
x0 = x_centre - (x1) - atom_spacing/2
z = 30

atoms = ase.Atoms(
    "Si1", positions=[(x_centre, 0, z)], cell=[x_centre*2, 0.1, 60]
)

phi_0 = 100e3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
abtem.show_atoms(atoms, ax=ax1, title="Beam view", numbering=True, merge=False)
abtem.show_atoms(atoms, ax=ax2, plane="xz", title="Side view", legend=True)

potential = abtem.Potential(atoms, sampling=0.1, projection="infinite", slice_thickness=30)

potential_array = potential.build().project().compute()



visualization = potential.show(
    project=True,
    explode=True,
    figsize=(16, 5),
    common_color_scale=True,
    cbar=True,
)

phi_0 = 100e3
plane_wave = abtem.PlaneWave(energy=100e3, extent=(10, 0.1), sampling=0.1)
# exit_wave = plane_wave.multislice(potential)
# exit_wave.compute()

potential_slice_at_sample = potential.build()[1]

transmission_function = potential_slice_at_sample.transmission_function(
    energy=100e3
)
propagator = abtem.multislice.FresnelPropagator()

transmitted_wave = transmission_function.transmit(plane_wave.build(), conjugate=False)
propagated_wave = propagator.propagate(transmitted_wave, thickness=30)

propagated_wave.intensity().show()
plt.show()
# intensity = exit_wave.intensity().compute()
# complex_image = exit_wave.complex_images().compute()
