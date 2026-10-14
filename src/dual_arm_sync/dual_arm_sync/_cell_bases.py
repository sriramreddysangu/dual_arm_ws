#!/usr/bin/env python3

import os

_CELL_BASES = {
    'dual': {
        'dsr01': (0.0,  0.695, 0.0),
        'dsr02': (0.0, -0.695, 0.0),
    },
    'four': {
        'dsr01': (0.0,   0.695, 0.0),
        'dsr02': (0.0,  -0.695, 0.0),
        'dsr03': (1.39,  0.695, 0.0),
        'dsr04': (1.39, -0.695, 0.0),
    },
    'six': {
        'dsr01': (0.0,   0.695, 0.0),
        'dsr02': (0.0,  -0.695, 0.0),
        'dsr03': (1.39,  0.695, 0.0),
        'dsr04': (1.39, -0.695, 0.0),
        'dsr05': (-1.39, 0.695, 0.0),
        'dsr06': (-1.39,-0.695, 0.0),
    },
    
    'nine': {
        'dsr01': (0.0,   0.695, 0.0),
        'dsr02': (0.0,  -0.695, 0.0),
        'dsr03': (1.39,  0.695, 0.0),
        'dsr04': (1.39, -0.695, 0.0),
        'dsr05': (-1.39, 0.695, 0.0),
        'dsr06': (-1.39,-0.695, 0.0),
        'dsr07': (0.0,   0.0,   0.0),
        'dsr08': (1.39,  0.0,   0.0),
        'dsr09': (-1.39, 0.0,   0.0),
    },
}

_CELLS = {
    'dual': ['dsr01', 'dsr02'],
    'four': ['dsr01', 'dsr02', 'dsr03', 'dsr04'],
    'six':  ['dsr01', 'dsr02', 'dsr03', 'dsr04', 'dsr05', 'dsr06'],
    'nine': ['dsr01', 'dsr02', 'dsr03', 'dsr04', 'dsr05', 'dsr06',
             'dsr07', 'dsr08', 'dsr09'],
}


def select_cell(name=None):
    """Return (ARM_NAMES, ROBOT_BASES) for a cell. `name` overrides the
    DUAL_ARM_CELL environment variable; defaults to 'six' if neither is set,
    matching the original pipeline's default ensemble."""
    key = (name or os.environ.get('DUAL_ARM_CELL', 'six')).strip().lower()
    if key not in _CELLS:
        raise ValueError(f"DUAL_ARM_CELL must be one of {sorted(_CELLS)}, got {key!r}")
    names = list(_CELLS[key])
    bases = dict(_CELL_BASES[key])
    return names, bases


ARM_NAMES, ROBOT_BASES = select_cell()