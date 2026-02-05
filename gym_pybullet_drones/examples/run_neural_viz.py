#!/usr/bin/env python
"""
Script launcher para neural_viz_real.py
Resuelve conflictos de OpenMP automáticamente
"""
import os
import sys
import subprocess

# Solucionar conflicto de OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Importar después de configurar el environment
import neural_viz_real

if __name__ == '__main__':
    neural_viz_real.main()
