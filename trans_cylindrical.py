# -*- coding: utf-8 -*-

# Calculating transmittance with an absorbing material

import math
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import sys

# TODO: this current simulation could be made much more efficient by taking
# advantage of cylindrical symmetry, effectively becoming a 2D simulation

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
    resolution = 100
    # Wavelength in microns
    wavelength = 1.55
    # perfectly matched layer thickness (microns)
    pml = 2
    # phi mode of the field
    m = 1

    # Geometry (units all microns)
    # Spacing between edge of simulation and source
    src_offset_z = 0.5
    # Spacing between source and start of absorber
    mosi_offset_z = 0.1
    mosi_length = 5
    mosi_center_z = src_offset_z / 2

    fiber_diameter = 1
    fiber_radius = fiber_diameter / 2
    mosi_real_thickness = 0.01 # 10 nm
    mosi_sim_thickness = 0.06 # 10 nm
    pad_thickness = 4

    fiber_index = 1.444

    # MoSi is ~1 times thicker than in reality to have enough simulation pixels
    # so we reduce its absorption by a factor of 1 to compensate
    mosi_thickness_comp = mosi_sim_thickness / mosi_real_thickness
    # Also compensate the difference in index by the same amount
    mosi_index = (1.61 - fiber_index) / mosi_thickness_comp + fiber_index
    mosi_k = 7.55
    conductivity = 2 * math.pi * wavelength * mosi_k / mosi_index / mosi_thickness_comp

    # Generate simulation obejcts
    dimensions = mp.CYLINDRICAL
    cell_r = fiber_radius + mosi_sim_thickness + pad_thickness
    cell_phi = 0
    cell_z = mosi_length + 2 * mosi_offset_z + 2 * src_offset_z + 2 * pml

    cell = mp.Vector3(cell_r, cell_phi, cell_z)

    duration = 10 * cell_z

    freq = 1/wavelength
    dfreq = freq/100

    src_pt = mp.Vector3(-cell_z/2 + pml + src_offset_z, 0, 0)

    output_slice = mp.Volume(center=mp.Vector3(), size=(cell_r, 0, cell_z))

    # Log important quantities
    print('File prefix: {}'.format(file_prefix))
    print('Duration: {}'.format(duration))
    print('Resolution: {}'.format(resolution))
    print('Dimensions (r, phi, z): {} um, {} um, {} um'.format(cell_r, cell_phi, cell_z))
    print('Wavelength: {} um'.format(wavelength))
    print('Fiber thickness: {} um'.format(fiber_diameter))
    print('Absorber dimensions: {} um, {} um'.format(mosi_length, mosi_real_thickness))
    print('Absorber center: {} um, {} um, {} um'.format(0, 0, 0))
    print('Absorber n: {}, k: {}'.format(mosi_index, mosi_k))
    print('Absorber compensation for thickness: {}'.format(mosi_thickness_comp))
    print('\n\n**********\n\n')

    default_material=mp.Medium(epsilon=1)

    # Physical geometry of the simulation
    geometry = [mp.Block(center=mp.Vector3(fiber_radius/2, 0, 0),
                         size=mp.Vector3(fiber_radius, mp.inf, mp.inf),
                         material=mp.Medium(epsilon=fiber_index))
                ]

    # Absorber will only be appended to geometry for the second simulation
    absorber = mp.Block(center=mp.Vector3(fiber_radius + mosi_sim_thickness / 2),
                        size=mp.Vector3(mosi_sim_thickness, mp.inf, mosi_length),
                        material=mp.Medium(epsilon=mosi_index, D_conductivity=conductivity))

    # Calculate eigenmode source
    sources = [mp.Source(mp.ContinuousSource(frequency=freq, width=0.1),
                     component=mp.Ez,
                     center=mp.Vector3(0, 0, -mosi_length / 2 - mosi_offset_z),
                     # size=mp.Vector3(3*fiber_radius)
                     ),

               # mp.Source(mp.ContinuousSource(freq, width=0.1),
               #           component=mp.Ep,
               #           center=mp.Vector3(0, 0, -mosi_length / 2 - mosi_offset_z),
               #           size=mp.Vector3(3*fiber_radius),
               #           amplitude=-1j)
           ]

    sources = [mp.Source(mp.ContinuousSource(frequency=freq),
                     component=mp.Ez,
                     size=mp.Vector3(3 * fiber_radius, mp.inf, 0),
                     center=mp.Vector3(0, 0, -mosi_length / 2 - mosi_offset_z))]

    # PML is the boundary layer around the edges of the simulation volume
    pml_layers = [mp.PML(pml)]

    # Pass all simulation parameters to meep
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        eps_averaging=True,
                        default_material=default_material,
                        dimensions=dimensions,
                        m=m)

    # Create flux monitors to calculate transmission and absorption
    fr_r = max(min(2 * fiber_radius, cell_r - 2 * pml), 0)

    # Reflected flux region
    refl_fr = mp.FluxRegion(center=mp.Vector3(0, 0, -0.5 * cell_z + pml + 2 * src_offset_z),
                            size=mp.Vector3(fr_r, mp.inf, 0))
    refl = sim.add_flux(freq, 0, 1, refl_fr)

    # Transmitted flux region
    tran_fr = mp.FluxRegion(center=mp.Vector3(0, 0, 0.5 * cell_z - pml - src_offset_z),
                            size=mp.Vector3(fr_r, mp.inf, 0))
    tran = sim.add_flux(freq, 0, 1, tran_fr)

    # Run simulation, outputting the epsilon distribution and the power in the
    # x-y plane every 0.25 microns/c
    sim.run(mp.at_beginning(mp.output_epsilon),
            # mp.to_appended("pwr",
            #                mp.in_volume(output_slice,
            #                             mp.at_every(0.05, mp.synchronized_magnetic(mp.output_tot_pwr))
            #                             )
            #                ),
            until=duration)

    print('\n\n**********\n\n')

    # For normalization run, save flux fields data for reflection plane
    no_absorber_refl_data = sim.get_flux_data(refl)
    # Save incident power for transmission plane
    no_absorber_tran_flux = mp.get_fluxes(tran)

    print("Flux: {}".format(no_absorber_tran_flux[0]))

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_r, 0, cell_z), component=mp.Dielectric)
    eps_cross_data = sim.get_array(center=mp.Vector3(x=cell_z/4), size=mp.Vector3(cell_r, cell_phi, 0), component=mp.Dielectric)

    # TODO: eliminate magic number
    max_field = 2

    # Plot epsilon distribution
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.axis('off')
        plt.savefig(file_prefix + '_Eps_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Eps_A.png')

    # Plot field on x-y plane
    ez_data = np.abs(sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_r, 0, cell_z), component=mp.Ez))
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Ez_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Ez_A.png')

    energy_side_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_r, 0, cell_z), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(energy_side_data.transpose(), interpolation='spline36', cmap='hot', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Pwr0_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr0_A.png')

    # Plot energy density on y-z plane
    energy_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_r, cell_phi, 0), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.plot(eps_cross_data)
        plt.plot(energy_data)
        # plt.axis('off')
        plt.savefig(file_prefix + '_Pwr1_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr1_A.png')

    energy_data = sim.get_array(center=mp.Vector3(z=cell_z/4), size=mp.Vector3(cell_r, cell_phi, 0), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.plot(eps_cross_data)
        plt.plot(energy_data)
        # plt.axis('off')
        plt.savefig(file_prefix + '_Pwr2_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr2_A.png')

    # Plot cross-sectional fields at several locations to ensure seeing nonzero fields
    num_x = 4
    num_y = 3
    fig, ax = plt.subplots(num_x, num_y)
    fig.suptitle('Cross Sectional Ez Fields')
    for i in range(num_x * num_y):
        monitor_z = i * 1/12 * cell_z / 2
        ez_cross_data = sim.get_array(center=mp.Vector3(z=monitor_z), size=mp.Vector3(cell_r, cell_phi, 0), component=mp.Ez)
        ax_num = i // num_y, i % num_y
        if mp.am_master():
            ax_eps = ax[ax_num].twinx()
            ax_eps.plot(eps_cross_data)
            ax[ax_num].plot(ez_cross_data)
            # ax[ax_num].axis('off')
            ax[ax_num].set_title('x = {}'.format(round(cell_z/4 + i / resolution, 3)))
    if mp.am_master():
        fig.tight_layout()
        plt.savefig(file_prefix + '_Ez_CS_A.png', dpi=300)
        print('Saved ' + file_prefix + '_Ez_CS_A.png')

    print('\n\n**********\n\n')

    # Reset simulation for absorption run
    sim.reset_meep()

    # Add the absorber material as the first item in the geometry list. It will
    # then be partially overwritten by the fiber object, giving the correct end
    # result
    geometry.insert(0, absorber)

    # Pass all simulation parameters to meep
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        eps_averaging=True,
                        default_material=default_material,
                        dimensions=dimensions,
                        m=m)

    refl = sim.add_flux(freq, 0, 1, refl_fr)
    tran = sim.add_flux(freq, 0, 1, tran_fr)

    sim.load_minus_flux_data(refl, no_absorber_refl_data)

    # Run simulation with absorber
    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.to_appended("pwr",
                           mp.in_volume(output_slice,
                                        mp.at_every(0.05, mp.synchronized_magnetic(mp.output_tot_pwr)))),
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

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_r, 0, cell_z), component=mp.Dielectric)
    eps_cross_data = sim.get_array(center=mp.Vector3(z=cell_z/4), size=mp.Vector3(cell_r, cell_phi, 0), component=mp.Dielectric)

    # Plot epsilon distribution
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.axis('off')
        plt.savefig(file_prefix + '_Eps_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Eps_B.png')

    # Plot field on x-y plane
    ez_data = np.abs(sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_r, 0, cell_z), component=mp.Ez))
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Ez_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Ez_B.png')

    energy_side_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_r, 0, cell_z), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(energy_side_data.transpose(), interpolation='spline36', cmap='hot', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Pwr0_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr0_B.png')

    # Plot energy density on y-z plane
    energy_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_r, cell_phi, 0), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.plot(eps_cross_data)
        plt.plot(energy_data)
        # plt.axis('off')
        plt.savefig(file_prefix + '_Pwr1_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr1_B.png')

    energy_data = sim.get_array(center=mp.Vector3(z=cell_z/4), size=mp.Vector3(cell_r, cell_phi, 0), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.plot(eps_cross_data)
        plt.plot(energy_data)
        # plt.axis('off')
        plt.savefig(file_prefix + '_Pwr2_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr2_B.png')

    # Plot cross-sectional fields at several locations to ensure seeing nonzero fields
    num_x = 4
    num_y = 3
    fig, ax = plt.subplots(num_x, num_y)
    fig.suptitle('Cross Sectional Ez Fields')
    for i in range(num_x * num_y):
        monitor_z = i * 1/12 * cell_z / 2
        ez_cross_data = sim.get_array(center=mp.Vector3(z=monitor_z), size=mp.Vector3(cell_r, cell_phi, 0), component=mp.Ez)
        ax_num = i // num_y, i % num_y
        if mp.am_master():
            ax[ax_num].plot(eps_cross_data)
            ax[ax_num].plot(ez_cross_data)
            ax[ax_num].axis('off')
            ax[ax_num].set_title('x = {}'.format(round(cell_z/4 + i / resolution, 3)))
    if mp.am_master():
        plt.savefig(file_prefix + '_Ez_CS_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Ez_CS_B.png')

    print('\n\n**********\n\n')
    print('Program finished.')
