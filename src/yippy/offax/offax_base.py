"""Base class for all off-axis classes."""


class OffAxBase:
    """Base class for all off-axis classes."""

    def __init__(self, offax_psf):
        """Initialize the OffAxBase class."""
        pass

    def __call__(self, x, y):
        """Return the PSF at the given off-axis position."""
        raise NotImplementedError("This method must be implemented in a subclass.")
