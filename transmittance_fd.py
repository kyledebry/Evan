# -*- coding: utf-8 -*-

# Calculating transmittance with an absorbing material

import math
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

def main():
    duration = 100
    resolution = 5
    cell_x = 20
    cell_y = 30
    cell_z = 30
    pml = 1
    src_buffer = 1
    mosi_buffer = 1
    mosi_length = cell_x - 2 * pml - src_buffer - 2 * mosi_buffer
    mosi_center_x = src_buffer / 2
    wavelength = 1.55
    cladding_thickness = 125
    core_thickness = 8
    core_radius = core_thickness / 2
    cladding_min_thickness = 1
    cladding_min_radius = cladding_min_thickness + core_radius
    mosi_thickness = 1
    mosi_width = 2
    bottom_min = core_radius + mosi_thickness + 2
    axis_y = 3 * cell_y / 8 - pml - bottom_min
    cell = mp.Vector3(cell_x, cell_y, cell_z)
    freq = 1/wavelength
    src_pt = mp.Vector3(-cell_x/2 + pml + src_buffer, 0, 0)

    default_material=mp.Medium(epsilon=1.444)

    geometry = [mp.Cylinder(center=mp.Vector3(y=axis_y), height=mp.inf, radius=core_radius,
                            material=mp.Medium(epsilon=1.4475),
                            axis=mp.Vector3(1,0,0)),
                mp.Block(mp.Vector3(mp.inf, cell_y, mp.inf),
                         center=mp.Vector3(0, cell_y / 2 + axis_y + cladding_min_radius, 0),
                         material=mp.Medium(epsilon=1))
                ]

    absorber = mp.Block(mp.Vector3(mosi_length, mosi_thickness, mosi_width),
                        center=mp.Vector3(mosi_center_x, axis_y + cladding_min_radius + mosi_thickness / 2, 0),
                        material=mp.Medium(epsilon=1.61, D_conductivity=2*math.pi*wavelength*7.55/1.61))

    sources = [mp.EigenModeSource(src=mp.ContinuousSource(frequency=freq),
              center=src_pt,
              size=mp.Vector3(0, cell_y - 2 * pml, cell_z - 2 * pml),
              eig_match_freq=True,
              eig_parity=mp.ODD_Z)]

    pml_layers = [mp.PML(pml)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        eps_averaging=False,
                        default_material=default_material,
                        symmetries=[mp.Mirror(mp.Z)],
                        force_complex_fields=True)

    fr_y = max(min(cladding_thickness, cell_y - 2 * pml), 0)
    fr_z = max(min(cladding_thickness, cell_z - 2 * pml), 0)

    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5 * cell_x + pml + src_buffer + 1, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    refl = sim.add_flux(freq, 0, 1, refl_fr)

    tran_fr = mp.FluxRegion(center=mp.Vector3(0.5 * cell_x - pml - 1, 0, 0),
                            size=mp.Vector3(0, fr_y, fr_z))
    tran = sim.add_flux(freq, 0, 1, tran_fr)

    sim.init_sim()
    sim.solve_cw(L=10)

    eps_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Dielectric)
    plt.figure()
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.axis('off')
    plt.show()

    # for normalization run, save flux fields data for reflection plane
    no_absorber_refl_data = sim.get_flux_data(refl)
    # save incident power for transmission plane
    no_absorber_tran_flux = mp.get_fluxes(tran)

    sim.reset_meep()

    geometry.append(absorber)

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution,
                        eps_averaging=False,
                        default_material=default_material,
                        symmetries=[mp.Mirror(mp.Z)],
                        force_complex_fields=True)

    refl = sim.add_flux(freq, 0, 1, refl_fr)
    tran = sim.add_flux(freq, 0, 1, tran_fr)

    sim.load_minus_flux_data(refl, no_absorber_refl_data)

    sim.init_sim()
    sim.solve_cw(L=10)

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
    plt.figure()
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.axis('off')
    plt.show()

    cm = pltcolors.LinearSegmentedColormap.from_list(
            'em', [(0,0,1), (0,0,0), (1,0,0)])

    ez_data = sim.get_array(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0), component=mp.Ez)
    plt.figure()
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
    plt.axis('off')
    plt.show()
