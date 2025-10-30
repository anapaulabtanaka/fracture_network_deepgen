#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Define new magic command (for IPython).
"""

# -------------------------------------------------------------------------
#   Author: Julien Straubhaar
#   Year: 2024
#   Company: University of Neuchâtel
#
#   Copyright (c) 2024 Julien Straubhaar
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# -------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# define %%skip_if magic command (from https://kioku-space.com/en/jupyter-skip-execution/)
# usage: %%skip_if <expression>

from IPython.core.magic import register_cell_magic
from IPython import get_ipython

@register_cell_magic
def skip_if(line, cell):
    if eval(line):
        return
    get_ipython().run_cell(cell)
# ------------------------------------------------------------------------------
