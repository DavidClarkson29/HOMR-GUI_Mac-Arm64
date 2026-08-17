"""Compatibility launcher for HOMR's explicit CPU mode."""

from homr_gui import main

if __name__ == "__main__":
    main(use_accelerated_inference=False)
