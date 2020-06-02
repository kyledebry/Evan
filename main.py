# -*- coding: utf-8 -*-
"""Main file for running each simulation"""
# Import the file you want to run here
# Run with "python main.py file_prefix 2>&1 | tee file_prefix.log"
# E.g. for Sim8b, run "python main.py Sim8b 2>&1 | tee Sim8b.log"
# And it will generate output files starting with Sim8b*
import trans_tapered as sim

sim.main()
