#!/bin/sh

#rm -f *.vtu

PYTHONPATH=/home/ganzenmueller/git/gcg-general-repo/LAMMPS_PyTools/pizza-9Feb12/src
export PYTHONPATH
python tt.py dump.LAMMPS

#~/git/gcg-general-repo/dump2vtk_tris/dump2vtk_tris dump_tool.LAMMPS tool
