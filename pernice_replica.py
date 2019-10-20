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
    resolution = 200
    # Simulation volume (um)
    cell_x = 2.8
    cell_y = 4
    cell_z = 4.5
    # Refractive indicies
    index_si = 3.6 # previously 3.4467
    index_sio2 = 1.444
    # Durations in units of micron/c
    duration = round(1.5 * cell_x + 2)
    num_timesteps = duration * resolution
    # Absorbing layer on boundary
    pml = 0.8
    # Geometry
    src_buffer = pml / 16
    nbn_buffer = src_buffer
    nbn_length = cell_x - 2 * pml - src_buffer - nbn_buffer
    nbn_center_x = (src_buffer + nbn_buffer) / 2
    wavelength = 1.55
    waveguide_width = 0.750 # 750 nm
    waveguide_height = 0.110 # 110 nm
    plane_shift_y = 0

    nbn_thickness = 0.010 # Actually 8 nm, but simulating 10 nm for 2 grid points
    nbn_width = 0.100 # 100 nm
    nbn_spacing = 0.120 # 120 nm

    # nbn is 10/8 times thicker than in reality to have enough simulation pixels
    # so we reduce its absorption by a factor of 5/4 to compensate
    nbn_thickness_comp = 5/4
    # Also compensate the difference in index by the same amount
    nbn_base_index = 5.23 # Taken from Hu thesis p86
    nbn_index = (5.23 - index_si) / nbn_thickness_comp + index_si
    nbn_base_k = 5.82 # Taken from Hu thesis p86
    nbn_k = nbn_base_k / nbn_thickness_comp
    conductivity = 2 * math.pi * wavelength * nbn_k / nbn_index

    flux_length = cell_x - 2 * pml - 4 * src_buffer

    # Generate simulation obejcts
    cell = mp.Vector3(cell_x, cell_y, cell_z)
    freq = 1/wavelength
    src_pt = mp.Vector3(-cell_x/2 + pml + src_buffer, 0, 0)
    output_slice = mp.Volume(center=mp.Vector3(y=(3 * waveguide_height / 4)), size=(cell_x, 0, cell_z))

    # Log important quantities
    print('NON-ABSORBING RUN')
    print('File prefix: {}'.format(file_prefix))
    print('Duration: {}'.format(duration))
    print('Resolution: {}'.format(resolution))
    print('Dimensions: {} um, {} um, {} um'.format(cell_x, cell_y, cell_z))
    print('Wavelength: {} um'.format(wavelength))
    print('Si thickness: {} um'.format(waveguide_height))
    print('NbN thickness: {} um'.format(nbn_thickness))
    print('Si index: {}; SiO2 index: {}'.format(index_si, index_sio2))
    print('Absorber dimensions: {} um, {} um, {} um'.format(nbn_length, nbn_thickness, nbn_width))
    print('Absorber n (base value): {} ({}), k: {} ({})'.format(nbn_index, nbn_base_index, nbn_k, nbn_base_k))
    print('Absorber compensation for thickness: {}'.format(nbn_thickness_comp))
    print('Flux lenght: {} um'.format(flux_length))
    print('\n\n**********\n\n')

    default_material=mp.Medium(epsilon=1)

    # Physical geometry of the simulation
    geometry = [
                mp.Block(mp.Vector3(mp.inf, cell_y, mp.inf),
                         center=mp.Vector3(0, - cell_y / 2 + plane_shift_y, 0),
                         material=mp.Medium(epsilon=index_sio2)),
                mp.Block(mp.Vector3(mp.inf, waveguide_height, waveguide_width),
                         center=mp.Vector3(0, waveguide_height / 2 + plane_shift_y, 0),
                         material=mp.Medium(epsilon=index_si))
                ]

    # Absorber will only be appended to geometry for the second simulation
    absorber = [mp.Block(mp.Vector3(nbn_length, nbn_thickness, nbn_width),
                        center=mp.Vector3(nbn_center_x, waveguide_height + nbn_thickness / 2, nbn_spacing / 2),
                        material=mp.Medium(epsilon=nbn_index, D_conductivity=conductivity)),
                mp.Block(mp.Vector3(nbn_length, nbn_thickness, nbn_width),
                        center=mp.Vector3(nbn_center_x, waveguide_height + nbn_thickness / 2, - nbn_spacing / 2),
                        material=mp.Medium(epsilon=nbn_index, D_conductivity=conductivity)),
                ]


    # Calculate eigenmode source
    src_max_y = cell_y - 2 * pml - 2 * src_buffer
    src_max_z = cell_z - 2 * pml - 2 * src_buffer
    src_y = src_max_y # min(8 * waveguide_height, src_max_y)
    src_z = src_max_z # min(3 * waveguide_width, src_max_z)

    sources = [mp.EigenModeSource(src=mp.ContinuousSource(frequency=freq),
              center=mp.Vector3(-cell_x / 2 + pml + src_buffer, waveguide_height / 2 + plane_shift_y, 0),
              size=mp.Vector3(0, src_y, src_z),
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
    fr_y = cell_y - 2 * pml
    fr_z = cell_z - 2 * pml

    # Reflected flux
    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5 * cell_x + pml + 2 * src_buffer, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    refl = sim.add_flux(freq, 0, 1, refl_fr)

    # Transmitted flux
    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5 * cell_x - pml - 2 * src_buffer, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    tran = sim.add_flux(freq, 0, 1, tran_fr)

    # Run simulation, outputting the epsilon distribution and the fields in the
    # x-y plane every 0.25 microns/c
    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.to_appended("ez_z0",
                           mp.in_volume(output_slice,
                                        mp.at_every(2/resolution, mp.output_efield_z))),
            until=duration)

    print('\n\n**********\n\n')

    sim.fields.synchronize_magnetic_fields()

    # For normalization run, save flux fields data for reflection plane
    no_absorber_refl_data = sim.get_flux_data(refl)
    # Save incident power for transmission plane
    no_absorber_tran_flux = mp.get_fluxes(tran)

    print("Flux: {}".format(no_absorber_tran_flux[0]))

    eps_data = sim.get_array(center=mp.Vector3(z=(nbn_spacing + nbn_width) / 2), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Dielectric)
    eps_cross_data = sim.get_array(center=mp.Vector3(x=cell_x/4), size=mp.Vector3(0, cell_y, cell_z), component=mp.Dielectric)

    max_field = 1.5

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

    energy_side_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(energy_side_data.transpose(), interpolation='spline36', cmap='hot', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Pwr0_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr0_A.png')

    # Plot energy density on y-z plane
    energy_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(0, cell_y, cell_z), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_cross_data, interpolation='spline36', cmap='binary')
        plt.imshow(energy_data, interpolation='spline36', cmap='hot', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Pwr1_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr1_A.png')

    energy_data = sim.get_array(center=mp.Vector3(x=cell_x/4), size=mp.Vector3(0, cell_y, cell_z), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_cross_data, interpolation='spline36', cmap='binary')
        plt.imshow(energy_data, interpolation='spline36', cmap='hot', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Pwr2_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr2_A.png')

    # Plot cross-sectional fields at several locations to ensure seeing nonzero fields
    num_x = 4
    num_y = 3
    fig, ax = plt.subplots(num_x, num_y)
    fig.suptitle('Cross Sectional Ez Fields')

    for i in range(num_x * num_y):
        monitor_x = i * (cell_x / 4) / (num_x * num_y)
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

    fig_e, ax_e = plt.subplots(num_x, num_y)
    fig_e.suptitle('Cross Sectional Energy Density')

    for i in range(num_x * num_y):
        monitor_x = i * (cell_x / 4) / (num_x * num_y)
        energy_cross_data = sim.get_array(center=mp.Vector3(x=monitor_x), size=mp.Vector3(0, cell_y, cell_z), component=mp.EnergyDensity)
        ax_num = i // num_y, i % num_y
        if mp.am_master():
            ax_e[ax_num].imshow(eps_cross_data, interpolation='spline36', cmap='binary')
            ax_e[ax_num].imshow(energy_cross_data, interpolation='spline36', cmap='hot', alpha=0.9)
            ax_e[ax_num].axis('off')
            ax_e[ax_num].set_title('x = {}'.format(round(cell_x/4 + i / resolution, 3)))
    if mp.am_master():
        plt.savefig(file_prefix + '_Pwr_CS_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr_CS_A.png')

    print('\n\n**********\n\n')
    """
    # Reset simulation for absorption run
    sim.reset_meep()

    geometry += absorber

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
    penetration_depth = - nbn_length / math.log(transmittance)

    print('Flux: {}'.format(absorber_tran_flux[0]))
    print("Transmittance: %f" % transmittance)
    print("Reflectance: %f" % reflectance)
    print("Absorption: {} over {} um".format(absorption, nbn_length))
    print("lambda = {} mm".format(penetration_depth / 1000))

    eps_data = sim.get_array(center=mp.Vector3(z=(nbn_spacing + nbn_width) / 2), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Dielectric)

    max_field = 1

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
        monitor_x = cell_x/4 + i / resolution
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
    """
    print('Program finished.')
