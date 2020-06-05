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
    resolution = 50
    # Simulation volume (um)
    cell_x = 8
    cell_y = 9
    cell_z = 9
    # Refractive indicies
    index_fiber = 1.444
    index_cladding = 3.5
    # Durations in units of micron/c
    duration = 2 * cell_x # round(1.5 * cell_x + 5)
    # Absorbing layer on boundary
    pml = 0.9
    # Geometry
    src_buffer = 0.1
    mosi_buffer = 0.1
    mosi_length = cell_x - 2 * pml - src_buffer - 2 * mosi_buffer
    mosi_center_x = src_buffer / 2
    wavelength = 1.55
    fiber_thickness = 1
    cladding_thickness = 3
    middle_layer_thickness = 2
    mosi_thickness = 0.04 # 40 nm
    # Properties of the absorber
    mosi_center_y = 0
    fiber_center_y = 0
    # MoSi is ~5 times thicker than in reality to have enough simulation pixels
    # so we reduce its absorption by a factor of 1 to compensate
    mosi_thickness_comp = 10
    # Also compensate the difference in index by the same amount
    mosi_index = (1.61 - index_fiber) / mosi_thickness_comp + index_fiber
    mosi_k = 7.55
    conductivity = 2 * math.pi * wavelength * mosi_k / mosi_index / mosi_thickness_comp

    # Generate simulation obejcts
    cell = mp.Vector3(cell_x, cell_y, cell_z)
    freq = 1/wavelength
    src_pt = mp.Vector3(-cell_x/2 + pml + src_buffer, 0, 0)
    output_slice = mp.Volume(center=mp.Vector3(), size=(cell_x, cell_y, 0))

    # Log important quantities
    print('File prefix: {}'.format(file_prefix))
    print('Duration: {}'.format(duration))
    print('Resolution: {}'.format(resolution))
    print('Dimensions: {} um, {} um, {} um'.format(cell_x, cell_y, cell_z))
    print('Wavelength: {} um'.format(wavelength))
    print('Fiber thickness: {} um'.format(fiber_thickness))
    print('Absorber dimensions: {} um, {} um'.format(mosi_length, mosi_thickness))
    print('Absorber center: {} um, {} um, {} um'.format(mosi_center_x, mosi_center_y, 0))
    print('Absorber n: {}, k: {}'.format(mosi_index, mosi_k))
    print('Absorber compensation for thickness: {}'.format(mosi_thickness_comp))
    print('\n\n**********\n\n')

    default_material=mp.Medium(epsilon=1)

    # Physical geometry of the simulation
    geometry = [mp.Cylinder(center=mp.Vector3(y=fiber_center_y), height=mp.inf, radius=cladding_thickness / 2,
                            material=mp.Medium(epsilon=index_cladding),
                            axis=mp.Vector3(1,0,0)),
                mp.Cylinder(center=mp.Vector3(y=fiber_center_y), height=mp.inf, radius=middle_layer_thickness / 2,
                            material=mp.Medium(epsilon=index_fiber),
                            axis=mp.Vector3(1,0,0)),
                mp.Cylinder(center=mp.Vector3(y=fiber_center_y), height=mp.inf, radius=fiber_thickness / 2,
                            material=mp.Medium(epsilon=index_fiber),
                            axis=mp.Vector3(1,0,0))
                ]

    # Absorber will only be appended to geometry for the second simulation
    absorber = mp.Cylinder(height=mosi_length, radius=fiber_thickness / 2 + mosi_thickness, axis=mp.Vector3(1,0,0),
                        center=mp.Vector3(mosi_center_x, mosi_center_y, 0),
                        material=mp.Medium(epsilon=mosi_index, D_conductivity=conductivity))

    # Calculate eigenmode source
    sources = [mp.EigenModeSource(src=mp.ContinuousSource(frequency=freq),
              center=mp.Vector3(-cell_x / 2 + pml + src_buffer, 0, 0),
              size=mp.Vector3(0, cell_y - 2 * pml, cell_z - 2 * pml),
              eig_match_freq=True,
              eig_parity=mp.ODD_Z,
              eig_band=1)]

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
                        symmetries=[mp.Mirror(mp.Z, phase=-1)])

    # Create flux monitors to calculate transmission and absorption
    fr_y = max(min(2 * fiber_thickness, cell_y - 2 * pml), 0)
    fr_z = max(min(2 * fiber_thickness, cell_z - 2 * pml), 0)

    # Reflected flux
    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5 * cell_x + pml + 2 * src_buffer, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    refl = sim.add_flux(freq, 0, 1, refl_fr)

    # Transmitted flux
    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5 * cell_x - pml - src_buffer, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    tran = sim.add_flux(freq, 0, 1, tran_fr)

    # Run simulation, outputting the epsilon distribution and the power in the
    # x-y plane every 0.25 microns/c
    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.to_appended("pwr",
                           mp.in_volume(output_slice,
                                        mp.at_every(0.05, mp.synchronized_magnetic(mp.output_tot_pwr)))),
            until=duration)

    print('\n\n**********\n\n')

    # For normalization run, save flux fields data for reflection plane
    no_absorber_refl_data = sim.get_flux_data(refl)
    # Save incident power for transmission plane
    no_absorber_tran_flux = mp.get_fluxes(tran)

    print("Flux: {}".format(no_absorber_tran_flux[0]))

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Dielectric)
    eps_cross_data = sim.get_array(center=mp.Vector3(x=cell_x/4), size=mp.Vector3(0, cell_y, cell_z), component=mp.Dielectric)

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
        monitor_x = i * 1/12 * cell_x / 2
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

    # Add the absorber material as the first item in the geometry list. It will
    # then be partially overwritten by the fiber object, giving the correct end
    # result
    geometry.insert(2, absorber)

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

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Dielectric)
    eps_cross_data = sim.get_array(center=mp.Vector3(x=cell_x/4), size=mp.Vector3(0, cell_y, cell_z), component=mp.Dielectric)

    # Plot epsilon distribution with absorber
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.axis('off')
        plt.savefig(file_prefix + '_Eps_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Eps_B.png')

    # Plot field on x-y plane
    ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Ez)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Ez_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Ez_B.png')

    energy_side_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(energy_side_data.transpose(), interpolation='spline36', cmap='hot', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Pwr0_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr0_B.png')

    # Plot energy density on y-z plane
    energy_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(0, cell_y, cell_z), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_cross_data, interpolation='spline36', cmap='binary')
        plt.imshow(energy_data, interpolation='spline36', cmap='hot', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Pwr1_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr1_B.png')

    energy_data = sim.get_array(center=mp.Vector3(x=cell_x/4), size=mp.Vector3(0, cell_y, cell_z), component=mp.EnergyDensity)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_cross_data, interpolation='spline36', cmap='binary')
        plt.imshow(energy_data, interpolation='spline36', cmap='hot', alpha=0.9)
        plt.axis('off')
        plt.savefig(file_prefix + '_Pwr2_B.png', dpi=300)
        print('Saved ' + file_prefix + '_Pwr2_B.png')

    # Plot field cross sections with absorber
    num_x = 4
    num_y = 3
    fig, ax = plt.subplots(num_x, num_y)
    fig.suptitle('Cross Sectional Ez Fields')
    for i in range(num_x * num_y):
        monitor_x = i * 1/12 * cell_x / 4
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
