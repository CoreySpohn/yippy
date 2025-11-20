"""Performance metric classes for coronagraph analysis."""

import numpy as np


class PerformanceMetric:
    """Class representing a coronagraph performance metric.

    This class provides a clean interface for accessing interpolated performance
    metrics at specific angular separations. It also allows direct indexing with
    angular separations.
    """

    def __init__(self, parent, metric_name, interpolator, unit="", description=""):
        """Initialize the performance metric.

        Args:
            parent (Coronagraph):
                The parent coronagraph object.
            metric_name (str):
                The name of the metric.
            interpolator:
                The spline interpolator for this metric.
            unit (str, optional):
                The unit string to display for this metric.
            description (str, optional):
                A longer description of what this metric represents.
        """
        self.parent = parent
        self.name = metric_name
        self.interpolator = interpolator
        self.unit = unit
        self.description = description

    def __call__(self, separation):
        """Allow calling the metric directly with a separation value.

        Args:
            separation (float, Quantity, or array-like):
                The separation(s) at which to evaluate the metric, in lambda/D.

        Returns:
            float or numpy.ndarray:
                The metric values at the specified separations.
        """
        return self.parent._interpolate_metric(separation, self.interpolator, 0.0)

    def __getitem__(self, separation):
        """Allow indexing the metric with a separation value.

        Args:
            separation (float, Quantity, or array-like):
                The separation(s) at which to evaluate the metric, in lambda/D.

        Returns:
            float or numpy.ndarray:
                The metric values at the specified separations.
        """
        return self.__call__(separation)

    def __repr__(self):
        """String representation of the performance metric."""
        return f"{self.name} ({self.unit})"

    def plot(
        self, separation_range=None, num_points=100, log_scale=False, ax=None, **kwargs
    ):
        """Plot the metric as a function of separation.

        Args:
            separation_range (tuple, optional):
                Tuple of (min, max) separation values in lambda/D.
                If None, use a reasonable range based on the coronagraph's IWA/OWA.
            num_points (int, optional):
                Number of points to plot. Default is 100.
            log_scale (bool, optional):
                Whether to use a logarithmic y-scale. Default is False.
            ax (matplotlib.axes.Axes, optional):
                Matplotlib axes to plot on. If None, creates a new figure.
            **kwargs:
                Additional keyword arguments to pass to matplotlib.pyplot.plot.

        Returns:
            tuple:
                Figure and Axes objects.
        """
        import matplotlib.pyplot as plt

        # Determine separation range if not provided
        if separation_range is None:
            if hasattr(self.parent, "IWA") and hasattr(self.parent, "offax"):
                # Use coronagraph's IWA as minimum and max off-axis separation
                # as maximum
                min_sep = max(
                    0.5,
                    self.parent.IWA.value
                    if hasattr(self.parent.IWA, "value")
                    else self.parent.IWA,
                )
                max_sep = max(self.parent.offax.x_offsets)
                separation_range = (min_sep, max_sep * 1.1)  # Add 10% margin
            else:
                # Default range if no IWA/OWA information
                separation_range = (1.0, 10.0)

        # Generate separation values and compute metric
        separations = np.linspace(separation_range[0], separation_range[1], num_points)
        values = self(separations)

        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        # Default plot settings
        plot_kwargs = {
            "linewidth": 2,
            "marker": "o",
            "markersize": 4,
            "label": self.name,
        }
        # Update with user-provided kwargs
        plot_kwargs.update(kwargs)

        # Plot the metric
        ax.plot(separations, values, **plot_kwargs)

        # Labels and title
        ax.set_xlabel("Separation [λ/D]")
        ax.set_ylabel(f"{self.name} {self.unit}")
        ax.set_title(f"{self.parent.name} {self.name}")

        # Apply log scale if requested
        if log_scale:
            ax.set_yscale("log")

        # Add grid
        ax.grid(True, alpha=0.3)

        return fig, ax

    def find_value(self, target_value, start_range=None, method="linear"):
        """Find the separation where the metric equals the target value.

        Args:
            target_value (float):
                The value to search for.
            start_range (tuple, optional):
                Initial search range as (min, max) separation in lambda/D.
                If None, use a default range.
            method (str, optional):
                The search method to use: 'linear' for linear interpolation,
                'scalar' for root finding. Default is 'linear'.

        Returns:
            float or None:
                The separation value where the metric equals the target_value,
                or None if no solution found.
        """
        from scipy.interpolate import interp1d
        from scipy.optimize import root_scalar

        # Default search range if not provided
        if start_range is None:
            if hasattr(self.parent, "IWA") and hasattr(self.parent, "offax"):
                min_sep = max(
                    0.5,
                    self.parent.IWA.value
                    if hasattr(self.parent.IWA, "value")
                    else self.parent.IWA,
                )
                max_sep = max(self.parent.offax.x_offsets)
                start_range = (min_sep, max_sep)
            else:
                start_range = (1.0, 10.0)

        if method == "linear":
            # Generate points and interpolate
            seps = np.linspace(start_range[0], start_range[1], 1000)
            vals = self(seps)
            interp = interp1d(vals, seps, bounds_error=False)
            result = (
                float(interp(target_value))
                if target_value >= min(vals) and target_value <= max(vals)
                else None
            )
        elif method == "scalar":
            # Define the objective function
            def objective(x):
                return self(x) - target_value

            # Try to find a root
            sol = root_scalar(objective, bracket=start_range)
            result = sol.root if sol.converged else None
        else:
            raise ValueError(f"Unknown method: {method}. Use 'linear' or 'scalar'.")

        return result

    def stats(self, sep_min=None, sep_max=None, num_points=100):
        """Calculate statistics for this metric within a separation range.

        Args:
            sep_min (float, optional):
                Minimum separation in lambda/D. If None, uses parent's IWA.
            sep_max (float, optional):
                Maximum separation in lambda/D. If None, uses max off-axis separation.
            num_points (int, optional):
                Number of evaluation points. Default is 100.

        Returns:
            dict:
                Dictionary containing statistics:
                - min: Minimum value
                - max: Maximum value
                - mean: Mean value
                - median: Median value
                - std: Standard deviation
                - separations: Separation values used
                - values: Metric values at each separation
        """
        # Determine separation range if not provided
        if sep_min is None and hasattr(self.parent, "IWA"):
            sep_min = (
                self.parent.IWA.value
                if hasattr(self.parent.IWA, "value")
                else self.parent.IWA
            )
        elif sep_min is None:
            sep_min = 1.0

        if sep_max is None and hasattr(self.parent, "offax"):
            sep_max = max(self.parent.offax.x_offsets)
        elif sep_max is None:
            sep_max = 10.0

        # Generate separation values and compute metric
        separations = np.linspace(sep_min, sep_max, num_points)
        values = self(separations)

        # Calculate statistics
        return {
            "min": np.min(values),
            "max": np.max(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "separations": separations,
            "values": values,
        }

    @staticmethod
    def compare(
        metrics, separation_range=None, num_points=100, log_scale=False, figsize=(10, 6)
    ):
        """Compare multiple performance metrics in a single plot.

        Args:
            metrics (list):
                List of PerformanceMetric objects to compare.
            separation_range (tuple, optional):
                Tuple of (min, max) separation values in lambda/D.
                If None, use automatic range determination.
            num_points (int, optional):
                Number of points to plot. Default is 100.
            log_scale (bool, optional):
                Whether to use a logarithmic y-scale. Default is False.
            figsize (tuple, optional):
                Figure size as (width, height). Default is (10, 6).

        Returns:
            tuple:
                Figure and Axes objects.
        """
        import matplotlib.pyplot as plt

        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)

        # Determine combined separation range if not provided
        if separation_range is None:
            min_seps = []
            max_seps = []

            for metric in metrics:
                if hasattr(metric.parent, "IWA") and hasattr(metric.parent, "offax"):
                    min_sep = max(
                        0.5,
                        metric.parent.IWA.value
                        if hasattr(metric.parent.IWA, "value")
                        else metric.parent.IWA,
                    )
                    max_sep = max(metric.parent.offax.x_offsets)
                    min_seps.append(min_sep)
                    max_seps.append(max_sep)

            if min_seps and max_seps:
                separation_range = (min(min_seps), max(max_seps) * 1.1)
            else:
                separation_range = (1.0, 10.0)

        # Generate separation values
        separations = np.linspace(separation_range[0], separation_range[1], num_points)

        # Plot each metric
        for i, metric in enumerate(metrics):
            values = metric(separations)

            # Use a different color for each metric
            ax.plot(
                separations,
                values,
                label=f"{metric.parent.name}: {metric.name}",
                marker="o",
                markersize=4,
                linewidth=2,
            )

        # Labels and title
        ax.set_xlabel("Separation [λ/D]")
        if all(metric.unit == metrics[0].unit for metric in metrics):
            ax.set_ylabel(f"{metrics[0].name} {metrics[0].unit}")
        else:
            ax.set_ylabel("Value")

        ax.set_title("Performance Metric Comparison")

        # Apply log scale if requested
        if log_scale:
            ax.set_yscale("log")

        # Add grid and legend
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        return fig, ax
