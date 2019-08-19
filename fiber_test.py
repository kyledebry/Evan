# -*- coding: utf-8 -*-

# From the Meep tutorial: plotting permittivity and fields of a straight waveguide
# Adapted for side polished fibers

import math
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors

def main():
    cell_x = 100
    cell_y = 150
    cell_z = 0
    pml = 1
    src_buffer = 2
    mosi_start_buffer = 2
    mosi_length = cell_x - 2 * pml - src_buffer - mosi_start_buffer
    mosi_center_x = (src_buffer + mosi_start_buffer) / 2
    wavelength = 1.55
    core_thickness = 8
    core_radius = core_thickness / 2
    cladding_min_thickness = 1
    cladding_min_radius = cladding_min_thickness + core_radius
    mosi_thickness = 1
    mosi_width = 2
    cell = mp.Vector3(cell_x, cell_y, cell_z)
    freq = 1/wavelength
    src_pt = mp.Vector3(-cell_x/2 + pml + src_buffer, 0, 0)

    geometry = [mp.Cylinder(center=mp.Vector3(), height=mp.inf, radius=125/2,
                            material=mp.Medium(epsilon=1.444),
                            axis=mp.Vector3(1,0,0)),
                mp.Cylinder(center=mp.Vector3(), height=mp.inf, radius=core_radius,
                            material=mp.Medium(epsilon=1.4475),
                            axis=mp.Vector3(1,0,0)),
                mp.Block(mp.Vector3(mp.inf, cell_y / 2 - cladding_min_radius, mp.inf),
                         center=mp.Vector3(0, cell_y / 4 + cladding_min_radius / 2, 0),
                         material=mp.Medium()),
                mp.Block(mp.Vector3(mosi_length, mosi_thickness, mosi_width),
                         center=mp.Vector3(mosi_center_x, cladding_min_radius + mosi_thickness / 2, 0),
                         material=mp.Medium(epsilon=1.61, D_conductivity=2*math.pi*0.15*7.55/1.61))
                ]

    sources = [mp.EigenModeSource(src=mp.ContinuousSource(frequency=freq),
              center=src_pt,
              size=mp.Vector3(y=cell_y - 2 * pml),
              eig_match_freq=True,
              eig_parity=mp.ODD_Z)]

    pml_layers = [mp.PML(pml)]

    resolution = 5

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    sim.run(until=200)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    plt.figure()
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.axis('off')
    plt.show()

    cm = pltcolors.LinearSegmentedColormap.from_list(
            'em', [(0,0,1), (0,0,0), (1,0,0)])

    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    plt.figure()
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.imshow(ez_data.transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
    plt.axis('off')
    plt.show()
