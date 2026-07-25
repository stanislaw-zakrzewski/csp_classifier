"""
Standard Performance Graph Builder Wrapper
===========================================

Delegates to graph_tools/build_graphs.py for building performance graphs.
"""

import sys
import os

# Add graph_tools to import path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from graph_tools.build_graphs import main

if __name__ == "__main__":
    main()
