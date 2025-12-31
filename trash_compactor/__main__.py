"""
Entry point for running trash-compactor as a module.
Allows: python -m trash_compactor ...
"""

from .cli import main

if __name__ == '__main__':
    main()
