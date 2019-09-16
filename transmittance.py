# -*- coding: utf-8 -*-

# Calculating transmittance with an absorbing material

import math
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors


class ZeroNormalize(pltcolors.Normalize):
    def __init__(self, vmax=1, clip=False):
        vmin = -vmax
        pltcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, 0, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def main():
    duration = 240
    resolution = 6
    cell_x = 120
    cell_y = 70
    cell_z = 110
    pml = 2
    src_buffer = 2
    mosi_buffer = 2
    mosi_length = cell_x - 2 * pml - src_buffer - 2 * mosi_buffer
    mosi_center_x = src_buffer / 2
    wavelength = 1.55
    cladding_thickness = 100
    core_thickness = 8
    core_radius = core_thickness / 2
    cladding_min_thickness = 1
    cladding_min_radius = cladding_min_thickness + core_radius
    mosi_thickness = 0.5
    mosi_width = 2
    bottom_min = core_radius + mosi_thickness
    axis_y = 4 * cell_y / 10 - pml - bottom_min
    cell = mp.Vector3(cell_x, cell_y, cell_z)
    freq = 1/wavelength
    src_pt = mp.Vector3(-cell_x/2 + pml + src_buffer, axis_y / 2, 0)
    output_slice = mp.Volume(center=mp.Vector3(), size=(cell_x, cell_y, 0))

    default_material=mp.Medium(epsilon=1)

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

    absorber = mp.Block(mp.Vector3(mosi_length, mosi_thickness, mosi_width),
                        center=mp.Vector3(mosi_center_x, axis_y + cladding_min_radius + mosi_thickness / 2, 0),
                        material=mp.Medium(epsilon=1.61, D_conductivity=2*math.pi*wavelength*7.55/1.61/50))

    sources = [mp.EigenModeSource(src=mp.ContinuousSource(frequency=freq),
              center=mp.Vector3(-cell_x / 2 + pml + src_buffer, axis_y, 0),
              size=mp.Vector3(0, cell_y - 4 * pml, cell_z - 4 * pml),
              eig_match_freq=True,
              eig_parity=mp.ODD_Z,
              eig_band=1)]

    pml_layers = [mp.PML(pml)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        eps_averaging=False,
                        default_material=default_material,
                        symmetries=[mp.Mirror(mp.Z, phase=-1)])

    fr_y = max(min(cladding_thickness, cell_y - 2 * pml), 0)
    fr_z = max(min(cladding_thickness, cell_z - 2 * pml), 0)

    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5 * cell_x + pml + 2 * src_buffer, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    refl = sim.add_flux(freq, 0, 1, refl_fr)

    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5 * cell_x - pml - src_buffer, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    tran = sim.add_flux(freq, 0, 1, tran_fr)

    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.to_appended("ez_z0",
                           mp.in_volume(output_slice,
                                        mp.at_every(0.25, mp.output_efield_z))),
            until=duration)

    # for normalization run, save flux fields data for reflection plane
    no_absorber_refl_data = sim.get_flux_data(refl)
    # save incident power for transmission plane
    no_absorber_tran_flux = mp.get_fluxes(tran)

    print("transmitted: {}".format(no_absorber_tran_flux))

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Dielectric)

    max_field = 0.1

    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.axis('off')
        plt.savefig('Sim3_Eps_A.png', dpi=300)

    ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Ez)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
        plt.axis('off')
        plt.savefig('Sim3_Ez_A.png', dpi=300)

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
            # ax[i].show()
            print("Plotted 3-{}".format(i))
    if mp.am_master():
        plt.savefig('Sim3_Ez_CS_A.png', dpi=300)


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

    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.to_appended("ez_z0",
                           mp.in_volume(output_slice,
                                        mp.at_every(0.25, mp.output_efield_z))),
            until=duration)

    absorber_refl_flux = mp.get_fluxes(refl)
    absorber_tran_flux = mp.get_fluxes(tran)

    transmittance = absorber_tran_flux[0] / no_absorber_tran_flux[0]
    reflectance = absorber_refl_flux[0] / no_absorber_tran_flux[0]
    absorption = 1 - transmittance
    penetration_depth = - mosi_length / math.log(transmittance)

    print("Transmittance: %f" % transmittance)
    print("Reflectance: %f" % reflectance)
    print("Absorption: {} over {} um".format(absorption, mosi_length))
    print("lambda = {} mm".format(penetration_depth / 1000))

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Dielectric)

    max_field = 0.1

    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.axis('off')
        plt.savefig('Sim3_Eps_B.png', dpi=300)

    ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Ez)
    if mp.am_master():
        plt.figure()
        plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
        plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
        plt.axis('off')
        plt.savefig('Sim3_Ez_B.png', dpi=300)

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
            # ax[i].show()
            print("Plotted 3-{}".format(i))
    if mp.am_master():
        plt.savefig('Sim3_Ez_CS_B.png', dpi=300)
