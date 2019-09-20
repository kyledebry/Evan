# -*- coding: utf-8 -*-

# Calculating transmittance with an absorbing material

import math
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import sys


# Custom color map to plot fields where 0 stays at the center of the color map
# even if all field values are positive
class ZeroNormalize(pltcolors.Normalize):
    def __init__(self, vmax=1, clip=False):
        vmin = -vmax
        pltcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, 0, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def main():
    # Prefix all output files with the command line argument
    file_prefix = sys.argv[1]
    # Number of pixels per micron
    resolution = 3
    # Simulation volume (um)
    cell_x = 100
    cell_y = 60
    cell_z = 100
    # Refractive indicies
    index_clad = 1.444
    index_core = 1.501 # For NA of 0.410
    numerical_aperature = math.sqrt(index_core ** 2 - index_clad ** 2)
    # Durations in units of micron/c
    duration = 2 * cell_x # round(1.5 * cell_x + 30)
    # Absorbing layer on boundary
    pml = 2
    # Geometry
    src_buffer = 2
    mosi_buffer = 2
    mosi_length = cell_x - 2 * pml - src_buffer - 2 * mosi_buffer
    mosi_center_x = src_buffer / 2
    wavelength = 1.55
    cladding_thickness = 90
    core_thickness = 2.3
    core_radius = core_thickness / 2
    cladding_min_thickness = 0.2
    cladding_min_radius = cladding_min_thickness + core_radius
    mosi_thickness = 0.5
    mosi_width = 2
    # Offset the axis of the fiber to reduce simulation volume in side-polished
    # fiber simulations
    bottom_min = core_radius + mosi_thickness
    axis_y = 4 * cell_y / 10 - pml - bottom_min
    # Properties of the absorber
    mosi_center_y = axis_y + cladding_min_radius + mosi_thickness / 2
    # MoSi is ~50 times thicker than in reality to have enough simulation pixels
    # so we reduce its absorption by a factor of 50 to compensate
    mosi_thickness_comp = 50
    # Also compensate the difference in index by the same amount
    mosi_index = (1.61 - index_clad) / mosi_thickness_comp + index_clad
    mosi_k = 7.55
    conductivity = 2 * math.pi * wavelength * mosi_k / mosi_index / mosi_thickness_comp

    # Generate simulation obejcts
    cell = mp.Vector3(cell_x, cell_y, cell_z)
    freq = 1/wavelength
    src_pt = mp.Vector3(-cell_x/2 + pml + src_buffer, axis_y / 2, 0)
    output_slice = mp.Volume(center=mp.Vector3(), size=(cell_x, cell_y, 0))

    # Log important quantities
    print('File prefix: {}'.format(file_prefix))
    print('Duration: {}'.format(duration))
    print('Resolution: {}'.format(resolution))
    print('Dimensions: {} um, {} um, {} um'.format(cell_x, cell_y, cell_z))
    print('Wavelength: {} um'.format(wavelength))
    print('Distance from core to absorber: {} um'.format(cladding_min_thickness))
    print('Core thickness: {} um'.format(core_thickness))
    print('Cladding max thickness: {} um'.format(cladding_thickness))
    print('Core index: {}; cladding index: {}'.format(index_core, index_clad))
    print('Numerical aperature: {}'.format(numerical_aperature))
    print('Absorber dimensions: {} um, {} um, {} um'.format(mosi_length, mosi_thickness, mosi_width))
    print('Absorber center: {} um, {} um, {} um'.format(mosi_center_x, mosi_center_y, 0))
    print('Absorber n: {}, k: {}'.format(mosi_index, mosi_k))
    print('Absorber compensation for thickness: {}'.format(mosi_thickness_comp))
    print('For this run, a high NA fiber is being simulated. This is using the '
    + '0.410 NA and 2.3 um core diameter values found at '
    + 'https://www.nufern.com/pam/optical_fibers/988/UHNA7/, and assuming that '
    + 'the cladding index is 1.444 as in standard fiber.')
    print('\n\n**********\n\n')

    default_material=mp.Medium(epsilon=1)

    # Physical geometry of the simulation
    geometry = [mp.Cylinder(center=mp.Vector3(y=axis_y), height=mp.inf, radius=cladding_thickness / 2,
                            material=mp.Medium(epsilon=1.444),
                            axis=mp.Vector3(1,0,0)),
                mp.Cylinder(center=mp.Vector3(y=axis_y), height=mp.inf, radius=core_radius,
                            material=mp.Medium(epsilon=1.4475),
                            axis=mp.Vector3(1,0,0)),
                mp.Block(mp.Vector3(mp.inf, cell_y, mp.inf),
                         center=mp.Vector3(0, cell_y / 2 + axis_y + cladding_min_radius, 0),
                         material=mp.Medium(epsilon=1))
                ]

    # Have a non-absorbing object of the same index to find correct waveguide mode
    absorber_replica = mp.Block(mp.Vector3(mp.inf, mosi_thickness, mosi_width),
                                center=mp.Vector3(mosi_center_x, mosi_center_y, 0),
                                material=mp.Medium(epsilon=mosi_index))

    geometry.append(absorber_replica)

    # Absorber will only be appended to geometry for the second simulation
    absorber = mp.Block(mp.Vector3(mosi_length, mosi_thickness, mosi_width),
                        center=mp.Vector3(mosi_center_x, mosi_center_y, 0),
                        material=mp.Medium(epsilon=mosi_index, D_conductivity=conductivity))

    # Calculate eigenmode source
    sources = [mp.EigenModeSource(src=mp.ContinuousSource(frequency=freq),
              center=mp.Vector3(-cell_x / 2 + pml + src_buffer, axis_y, 0),
              size=mp.Vector3(0, cell_y - 4 * pml, cell_z - 4 * pml),
              eig_match_freq=True,
              eig_parity=mp.ODD_Z,
              eig_band=1)]

    pml_layers = [mp.PML(pml)]

    # Pass all simulation parameters to meep
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        eps_averaging=False,
                        default_material=default_material,
                        symmetries=[mp.Mirror(mp.Z, phase=-1)])

    # Create flux monitors to calculate transmission and absorption
    fr_y = max(min(cladding_thickness, cell_y - 2 * pml), 0)
    fr_z = max(min(cladding_thickness, cell_z - 2 * pml), 0)

    # Reflected flux
    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5 * cell_x + pml + 2 * src_buffer, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    refl = sim.add_flux(freq, 0, 1, refl_fr)

    # Transmitted flux
    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5 * cell_x - pml - src_buffer, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    tran = sim.add_flux(freq, 0, 1, tran_fr)

    # Run simulation, outputting the epsilon distribution and the fields in the
    # x-y plane every 0.25 microns/c
    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.to_appended("ez_z0",
                           mp.in_volume(output_slice,
                                        mp.at_every(0.25, mp.output_efield_z))),
            until=duration)

    print('\n\n**********\n\n')

    # For normalization run, save flux fields data for reflection plane
    no_absorber_refl_data = sim.get_flux_data(refl)
    # Save incident power for transmission plane
    no_absorber_tran_flux = mp.get_fluxes(tran)

    print("Flux: {}".format(no_absorber_tran_flux[0]))

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Dielectric)

    max_field = 0.1

    # Plot epsilon distribution
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.axis('off')
        plt.savefig(file_prefix + '_Eps_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Eps_A.png')

    # Plot field on x-y plane
    ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Ez)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Ez_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Ez_A.png')

    # Plot cross-sectional fields at several locations to ensure seeing nonzero fields
    eps_cross_data = sim.get_array(center=mp.Vector3(x=cell_x/4), size=mp.Vector3(0, cell_y, cell_z), component=mp.Dielectric)
    num_x = 4
    num_y = 3
    fig, ax = plt.subplots(num_x, num_y)
    fig.suptitle('Cross Sectional Ez Fields')
    for i in range(num_x * num_y):
        monitor_x = cell_x/4 + i / (wavelength / 1.444) / 4
        ez_cross_data = sim.get_array(center=mp.Vector3(x=monitor_x), size=mp.Vector3(0, cell_y, cell_z), component=mp.Ez)
        ax_num = i // num_y, i % num_y
        if mp.am_master():
            ax[ax_num].imshow(eps_cross_data, interpolation='spline36', cmap='binary')
            ax[ax_num].imshow(ez_cross_data, interpolation='spline36', cmap='RdBu', alpha=0.9, norm=ZeroNormalize(vmax=np.max(max_field)))
            ax[ax_num].axis('off')
            ax[ax_num].set_title('x = {}'.format(round(cell_x/4 + i / resolution, 3)))
    if mp.am_master():
        plt.savefig(file_prefix + '_Ez_CS_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Ez_CS_A.png')

    print('\n\n**********\n\n')

    # Reset simulation for absorption run
    sim.reset_meep()

    geometry.append(absorber)

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        eps_averaging=False,
                        default_material=default_material,
                        symmetries=[mp.Mirror(mp.Z, phase=-1)])

    refl = sim.add_flux(freq, 0, 1, refl_fr)
    tran = sim.add_flux(freq, 0, 1, tran_fr)

    sim.load_minus_flux_data(refl, no_absorber_refl_data)

    # Run simulation with absorber
    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.to_appended("ez_z0",
                           mp.in_volume(output_slice,
                                        mp.at_every(0.25, mp.output_efield_z))),
            until=duration)

    print('\n\n**********\n\n')

    # Calculate transmission and absorption
    absorber_refl_flux = mp.get_fluxes(refl)
    absorber_tran_flux = mp.get_fluxes(tran)

    transmittance = absorber_tran_flux[0] / no_absorber_tran_flux[0]
    reflectance = absorber_refl_flux[0] / no_absorber_tran_flux[0]
    absorption = 1 - transmittance
    penetration_depth = - mosi_length / math.log(transmittance)

    print('Flux: {}'.format(absorber_tran_flux[0]))
    print("Transmittance: %f" % transmittance)
    print("Reflectance: %f" % reflectance)
    print("Absorption: {} over {} um".format(absorption, mosi_length))
    print("lambda = {} mm".format(penetration_depth / 1000))

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Dielectric)

    max_field = 0.1

    # Plot epsilon distribution with absorber
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.axis('off')
        plt.savefig(file_prefix + '_Eps_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Eps_B.png')

    # Plot fields in x-y plane with absorber
    ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Ez)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Ez_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Ez_B.png')

    # Plot field cross sections with absorber
    eps_cross_data = sim.get_array(center=mp.Vector3(x=cell_x/4), size=mp.Vector3(0, cell_y, cell_z), component=mp.Dielectric)
    num_x = 4
    num_y = 3
    fig, ax = plt.subplots(num_x, num_y)
    fig.suptitle('Cross Sectional Ez Fields')
    for i in range(num_x * num_y):
        monitor_x = cell_x/4 + i / (wavelength / 1.444) / 4
        ez_cross_data = sim.get_array(center=mp.Vector3(x=monitor_x), size=mp.Vector3(0, cell_y, cell_z), component=mp.Ez)
        ax_num = i // num_y, i % num_y
        if mp.am_master():
            ax[ax_num].imshow(eps_cross_data, interpolation='spline36', cmap='binary')
            ax[ax_num].imshow(ez_cross_data, interpolation='spline36', cmap='RdBu', alpha=0.9, norm=ZeroNormalize(vmax=np.max(max_field)))
            ax[ax_num].axis('off')
            ax[ax_num].set_title('x = {}'.format(round(cell_x/4 + i / resolution, 3)))
    if mp.am_master():
        plt.savefig(file_prefix + '_Ez_CS_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Ez_CS_B.png')

    print('\n\n**********\n\n')
    print('Program finished.')
