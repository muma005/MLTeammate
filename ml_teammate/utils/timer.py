"""
Timer Utilities for MLTeammate

Provides performance monitoring and timing utilities for tracking
experiment execution times and performance metrics.
"""

import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime, timedelta


class Timer:
    """
    High-precision timer for performance monitoring.
    """
    
    def __init__(self, name: str = "Timer"):
        """
        Initialize timer with optional name.
        
        Args:
            name: Timer name for identification
        """
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: Optional[float] = None
        self.is_running = False
    
    def start(self):
        """Start the timer."""
        if self.is_running:
            raise RuntimeError(f"Timer '{self.name}' is already running")
        
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed_time = None
        self.is_running = True
    
    def stop(self) -> float:
        """
        Stop the timer and return elapsed time.
        
        Returns:
            float: Elapsed time in seconds
            
        Raises:
            RuntimeError: If timer is not running
        """
        if not self.is_running:
            raise RuntimeError(f"Timer '{self.name}' is not running")
        
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        self.is_running = False
        
        return self.elapsed_time
    
    def get_elapsed(self) -> float:
        """
        Get elapsed time (current or final).
        
        Returns:
            float: Elapsed time in seconds
        """
        if self.is_running:
            return time.perf_counter() - self.start_time
        elif self.elapsed_time is not None:
            return self.elapsed_time
        else:
            return 0.0
    
    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
        self.is_running = False
    
    def restart(self):
        """Reset and start the timer."""
        self.reset()
        self.start()
    
    def __enter__(self):
        """Enter context manager protocol."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager protocol."""
        if self.is_running:
            self.stop()
        return False  # Don't suppress exceptions
    
    def __str__(self) -> str:
        """String representation of timer."""
        if self.is_running:
            return f"Timer '{self.name}': Running ({self.get_elapsed():.3f}s)"
        elif self.elapsed_time is not None:
            return f"Timer '{self.name}': {self.elapsed_time:.3f}s"
        else:
            return f"Timer '{self.name}': Not started"


@contextmanager
def time_context(name: str = "Operation"):
    """
    Context manager for timing operations.
    
    Args:
        name: Operation name
        
    Yields:
        Timer: Timer instance
        
    Example:
        with time_context("Model Training") as timer:
            model.fit(X, y)
        print(f"Training took {timer.elapsed_time:.3f}s")
    """
    timer = Timer(name)
    timer.start()
    try:
        yield timer
    finally:
        if timer.is_running:
            timer.stop()


def time_function(func, *args, **kwargs) -> tuple:
    """
    Time a function call and return result with timing.
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        tuple: (result, elapsed_time)
    """
    timer = Timer(func.__name__)
    timer.start()
    
    try:
        result = func(*args, **kwargs)
        elapsed = timer.stop()
        return result, elapsed
    except Exception as e:
        timer.stop()
        raise e


class ExperimentTimer:
    """
    Advanced timer for tracking multiple phases of experiments.
    """
    
    def __init__(self, experiment_name: str = "Experiment"):
        """
        Initialize experiment timer.
        
        Args:
            experiment_name: Name of the experiment
        """
        self.experiment_name = experiment_name
        self.phase_timers: Dict[str, Timer] = {}
        self.phase_history: Dict[str, list] = {}
        self.experiment_start: Optional[float] = None
        self.experiment_end: Optional[float] = None
    
    def start_experiment(self):
        """Start timing the entire experiment."""
        self.experiment_start = time.perf_counter()
        self.experiment_end = None
    
    def end_experiment(self) -> float:
        """
        End experiment timing.
        
        Returns:
            float: Total experiment time
        """
        if self.experiment_start is None:
            raise RuntimeError("Experiment not started")
        
        self.experiment_end = time.perf_counter()
        return self.experiment_end - self.experiment_start
    
    def start_phase(self, phase_name: str):
        """
        Start timing a phase.
        
        Args:
            phase_name: Name of the phase
        """
        if phase_name in self.phase_timers and self.phase_timers[phase_name].is_running:
            raise RuntimeError(f"Phase '{phase_name}' is already running")
        
        timer = Timer(phase_name)
        timer.start()
        self.phase_timers[phase_name] = timer
    
    def end_phase(self, phase_name: str) -> float:
        """
        End timing a phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            float: Phase elapsed time
        """
        if phase_name not in self.phase_timers:
            raise RuntimeError(f"Phase '{phase_name}' not started")
        
        elapsed = self.phase_timers[phase_name].stop()
        
        # Add to history
        if phase_name not in self.phase_history:
            self.phase_history[phase_name] = []
        self.phase_history[phase_name].append(elapsed)
        
        return elapsed
    
    def get_phase_time(self, phase_name: str) -> Optional[float]:
        """
        Get the time for a specific phase.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            float or None: Phase time if available
        """
        if phase_name in self.phase_timers:
            return self.phase_timers[phase_name].get_elapsed()
        return None
    
    def get_phase_statistics(self, phase_name: str) -> Dict[str, float]:
        """
        Get statistics for a phase that ran multiple times.
        
        Args:
            phase_name: Name of the phase
            
        Returns:
            dict: Statistics (mean, min, max, total)
        """
        if phase_name not in self.phase_history:
            return {}
        
        times = self.phase_history[phase_name]
        if not times:
            return {}
        
        return {
            "count": len(times),
            "total": sum(times),
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times)
        }
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive experiment timing summary.
        
        Returns:
            dict: Experiment summary
        """
        summary = {
            "experiment_name": self.experiment_name,
            "total_time": None,
            "phases": {}
        }
        
        # Total experiment time
        if self.experiment_start is not None:
            if self.experiment_end is not None:
                summary["total_time"] = self.experiment_end - self.experiment_start
            else:
                summary["total_time"] = time.perf_counter() - self.experiment_start
        
        # Phase summaries
        for phase_name in self.phase_timers:
            phase_stats = self.get_phase_statistics(phase_name)
            current_time = self.get_phase_time(phase_name)
            
            summary["phases"][phase_name] = {
                "current_time": current_time,
                "statistics": phase_stats
            }
        
        return summary
    
    def format_summary(self) -> str:
        """
        Format experiment summary as readable string.
        
        Returns:
            str: Formatted summary
        """
        summary = self.get_experiment_summary()
        
        lines = [f"Experiment: {summary['experiment_name']}"]
        
        if summary["total_time"] is not None:
            lines.append(f"Total Time: {summary['total_time']:.3f}s")
        
        lines.append("\nPhase Breakdown:")
        
        for phase_name, phase_data in summary["phases"].items():
            current = phase_data["current_time"]
            stats = phase_data["statistics"]
            
            if current is not None:
                lines.append(f"  {phase_name}: {current:.3f}s")
            
            if stats and stats["count"] > 1:
                lines.append(f"    └─ Avg: {stats['mean']:.3f}s ({stats['count']} runs)")
        
        return "\n".join(lines)


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"


def estimate_remaining_time(current_iteration: int, total_iterations: int, 
                          elapsed_time: float) -> Optional[float]:
    """
    Estimate remaining time based on current progress.
    
    Args:
        current_iteration: Current iteration number (0-based)
        total_iterations: Total number of iterations
        elapsed_time: Time elapsed so far
        
    Returns:
        float or None: Estimated remaining time in seconds
    """
    if current_iteration <= 0 or total_iterations <= 0:
        return None
    
    if current_iteration >= total_iterations:
        return 0.0
    
    avg_time_per_iteration = elapsed_time / current_iteration
    remaining_iterations = total_iterations - current_iteration
    
    return avg_time_per_iteration * remaining_iterations
