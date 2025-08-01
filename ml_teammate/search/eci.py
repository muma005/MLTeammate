"""
Early Convergence Indicator (ECI) for MLTeammate

This module provides early convergence detection for hyperparameter
optimization, helping to stop the search process when further trials
are unlikely to improve results significantly.

ECI uses various statistical methods to detect convergence:
- Moving average analysis
- Improvement rate analysis
- Confidence interval analysis
- Plateau detection
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from scipy import stats
import warnings


class EarlyConvergenceIndicator:
    """
    Early Convergence Indicator for hyperparameter optimization.
    
    Detects when the optimization process has converged and further
    trials are unlikely to provide significant improvements.
    """
    
    def __init__(self,
                 window_size: int = 10,
                 min_trials: int = 5,
                 improvement_threshold: float = 0.001,
                 confidence_level: float = 0.95,
                 patience: int = 5,
                 convergence_method: str = "moving_average"):
        """
        Initialize the Early Convergence Indicator.
        
        Args:
            window_size: Size of the moving window for analysis
            min_trials: Minimum number of trials before checking convergence
            improvement_threshold: Minimum improvement threshold
            confidence_level: Confidence level for statistical tests
            patience: Number of consecutive non-improving trials before stopping
            convergence_method: Method to use for convergence detection
        """
        self.window_size = window_size
        self.min_trials = min_trials
        self.improvement_threshold = improvement_threshold
        self.confidence_level = confidence_level
        self.patience = patience
        self.convergence_method = convergence_method
        
        # Trial history
        self.scores = []
        self.timestamps = []
        self.best_score = None
        self.best_trial = None
        
        # Convergence tracking
        self.consecutive_no_improvement = 0
        self.converged = False
        self.convergence_reason = None
        
        # Method-specific parameters
        self.moving_averages = deque(maxlen=window_size)
        self.improvement_rates = deque(maxlen=window_size)
    
    def add_trial(self, trial_id: str, score: float, timestamp: Optional[float] = None):
        """
        Add a trial result to the convergence analysis.
        
        Args:
            trial_id: Trial identifier
            score: Trial score
            timestamp: Trial timestamp (optional)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.scores.append(score)
        self.timestamps.append(timestamp)
        
        # Update best score
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_trial = trial_id
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1
        
        # Check for convergence if we have enough trials
        if len(self.scores) >= self.min_trials:
            self._check_convergence()
    
    def _check_convergence(self):
        """Check if the optimization has converged."""
        if self.converged:
            return
        
        n_trials = len(self.scores)
        
        if self.convergence_method == "moving_average":
            converged = self._check_moving_average_convergence()
        elif self.convergence_method == "improvement_rate":
            converged = self._check_improvement_rate_convergence()
        elif self.convergence_method == "confidence_interval":
            converged = self._check_confidence_interval_convergence()
        elif self.convergence_method == "plateau_detection":
            converged = self._check_plateau_convergence()
        elif self.convergence_method == "combined":
            converged = self._check_combined_convergence()
        else:
            converged = False
        
        # Check patience-based convergence
        patience_converged = self.consecutive_no_improvement >= self.patience
        
        if converged or patience_converged:
            self.converged = True
            if converged:
                self.convergence_reason = f"{self.convergence_method}_convergence"
            else:
                self.convergence_reason = "patience_exceeded"
    
    def _check_moving_average_convergence(self) -> bool:
        """Check convergence using moving average analysis."""
        if len(self.scores) < self.window_size:
            return False
        
        # Calculate moving average
        recent_scores = self.scores[-self.window_size:]
        moving_avg = np.mean(recent_scores)
        
        # Calculate standard deviation
        std_dev = np.std(recent_scores)
        
        # Check if the moving average is stable
        if std_dev < self.improvement_threshold:
            return True
        
        # Check if the moving average is close to the best score
        if abs(moving_avg - self.best_score) < self.improvement_threshold:
            return True
        
        return False
    
    def _check_improvement_rate_convergence(self) -> bool:
        """Check convergence using improvement rate analysis."""
        if len(self.scores) < self.window_size:
            return False
        
        # Calculate improvement rates
        improvements = []
        for i in range(1, len(self.scores)):
            improvement = self.scores[i] - self.scores[i-1]
            improvements.append(improvement)
        
        # Calculate recent improvement rate
        recent_improvements = improvements[-self.window_size:]
        avg_improvement = np.mean(recent_improvements)
        
        # Check if improvement rate is below threshold
        if avg_improvement < self.improvement_threshold:
            return True
        
        return False
    
    def _check_confidence_interval_convergence(self) -> bool:
        """Check convergence using confidence interval analysis."""
        if len(self.scores) < self.window_size:
            return False
        
        # Calculate confidence interval for recent scores
        recent_scores = self.scores[-self.window_size:]
        
        try:
            # Calculate confidence interval
            mean_score = np.mean(recent_scores)
            std_error = stats.sem(recent_scores)
            confidence_interval = stats.t.interval(
                self.confidence_level, 
                len(recent_scores) - 1, 
                loc=mean_score, 
                scale=std_error
            )
            
            # Check if confidence interval is narrow
            interval_width = confidence_interval[1] - confidence_interval[0]
            if interval_width < self.improvement_threshold:
                return True
            
            # Check if best score is within confidence interval
            if confidence_interval[0] <= self.best_score <= confidence_interval[1]:
                return True
                
        except (ValueError, RuntimeWarning):
            # Handle edge cases where statistical tests fail
            pass
        
        return False
    
    def _check_plateau_convergence(self) -> bool:
        """Check convergence using plateau detection."""
        if len(self.scores) < self.window_size:
            return False
        
        # Detect plateaus using linear regression
        recent_scores = self.scores[-self.window_size:]
        x = np.arange(len(recent_scores))
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_scores)
            
            # Check if slope is very small (plateau)
            if abs(slope) < self.improvement_threshold:
                return True
            
            # Check if R-squared is very low (no clear trend)
            if r_value**2 < 0.1:
                return True
                
        except (ValueError, RuntimeWarning):
            pass
        
        return False
    
    def _check_combined_convergence(self) -> bool:
        """Check convergence using multiple methods."""
        methods = [
            self._check_moving_average_convergence,
            self._check_improvement_rate_convergence,
            self._check_confidence_interval_convergence,
            self._check_plateau_convergence
        ]
        
        convergence_count = sum(1 for method in methods if method())
        
        # Require at least 2 methods to agree on convergence
        return convergence_count >= 2
    
    def should_stop(self) -> bool:
        """
        Check if the optimization should stop.
        
        Returns:
            True if optimization should stop, False otherwise
        """
        return self.converged
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """
        Get information about the convergence status.
        
        Returns:
            Dictionary with convergence information
        """
        info = {
            "converged": self.converged,
            "convergence_reason": self.convergence_reason,
            "n_trials": len(self.scores),
            "best_score": self.best_score,
            "best_trial": self.best_trial,
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "patience": self.patience
        }
        
        if len(self.scores) >= self.window_size:
            recent_scores = self.scores[-self.window_size:]
            info.update({
                "recent_mean": np.mean(recent_scores),
                "recent_std": np.std(recent_scores),
                "recent_min": np.min(recent_scores),
                "recent_max": np.max(recent_scores)
            })
        
        return info
    
    def get_improvement_curve(self) -> Dict[str, List[float]]:
        """
        Get the improvement curve data.
        
        Returns:
            Dictionary with improvement curve data
        """
        if len(self.scores) < 2:
            return {"trials": [], "scores": [], "improvements": []}
        
        trials = list(range(1, len(self.scores) + 1))
        improvements = [0]  # First trial has no improvement
        
        for i in range(1, len(self.scores)):
            improvement = self.scores[i] - self.scores[i-1]
            improvements.append(improvement)
        
        return {
            "trials": trials,
            "scores": self.scores.copy(),
            "improvements": improvements
        }
    
    def reset(self):
        """Reset the convergence indicator."""
        self.scores = []
        self.timestamps = []
        self.best_score = None
        self.best_trial = None
        self.consecutive_no_improvement = 0
        self.converged = False
        self.convergence_reason = None
        self.moving_averages.clear()
        self.improvement_rates.clear()


class AdaptiveECI(EarlyConvergenceIndicator):
    """
    Adaptive Early Convergence Indicator.
    
    This variant adapts its parameters based on the optimization progress
    and the characteristics of the objective function.
    """
    
    def __init__(self, **kwargs):
        """Initialize Adaptive ECI."""
        super().__init__(**kwargs)
        
        # Adaptive parameters
        self.adaptation_window = 20
        self.score_variance_history = deque(maxlen=self.adaptation_window)
        self.improvement_variance_history = deque(maxlen=self.adaptation_window)
    
    def add_trial(self, trial_id: str, score: float, timestamp: Optional[float] = None):
        """Add trial and adapt parameters."""
        super().add_trial(trial_id, score, timestamp)
        
        # Adapt parameters based on recent performance
        if len(self.scores) >= self.adaptation_window:
            self._adapt_parameters()
    
    def _adapt_parameters(self):
        """Adapt convergence parameters based on recent performance."""
        recent_scores = self.scores[-self.adaptation_window:]
        
        # Calculate score variance
        score_variance = np.var(recent_scores)
        self.score_variance_history.append(score_variance)
        
        # Calculate improvement variance
        improvements = np.diff(recent_scores)
        improvement_variance = np.var(improvements) if len(improvements) > 1 else 0
        self.improvement_variance_history.append(improvement_variance)
        
        # Adapt improvement threshold based on variance
        if len(self.score_variance_history) >= 5:
            avg_variance = np.mean(list(self.score_variance_history))
            self.improvement_threshold = max(0.0001, avg_variance * 0.1)
        
        # Adapt patience based on improvement pattern
        if len(self.improvement_variance_history) >= 5:
            avg_improvement_variance = np.mean(list(self.improvement_variance_history))
            if avg_improvement_variance < 0.001:
                self.patience = max(3, self.patience - 1)
            elif avg_improvement_variance > 0.01:
                self.patience = min(10, self.patience + 1)


class MultiObjectiveECI(EarlyConvergenceIndicator):
    """
    Multi-objective Early Convergence Indicator.
    
    This variant handles multi-objective optimization scenarios
    where multiple metrics need to be considered for convergence.
    """
    
    def __init__(self, objectives: List[str], **kwargs):
        """
        Initialize Multi-objective ECI.
        
        Args:
            objectives: List of objective names
            **kwargs: Additional arguments for EarlyConvergenceIndicator
        """
        super().__init__(**kwargs)
        self.objectives = objectives
        self.multi_scores = {obj: [] for obj in objectives}
        self.best_scores = {obj: None for obj in objectives}
    
    def add_trial(self, trial_id: str, scores: Dict[str, float], timestamp: Optional[float] = None):
        """
        Add a multi-objective trial result.
        
        Args:
            trial_id: Trial identifier
            scores: Dictionary of objective scores
            timestamp: Trial timestamp (optional)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Store scores for each objective
        for obj in self.objectives:
            if obj in scores:
                self.multi_scores[obj].append(scores[obj])
                
                # Update best score for this objective
                if self.best_scores[obj] is None or scores[obj] > self.best_scores[obj]:
                    self.best_scores[obj] = scores[obj]
        
        # Use a composite score for convergence analysis
        composite_score = self._calculate_composite_score(scores)
        super().add_trial(trial_id, composite_score, timestamp)
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """
        Calculate a composite score from multiple objectives.
        
        Args:
            scores: Dictionary of objective scores
            
        Returns:
            Composite score
        """
        # Simple weighted average (can be customized)
        weights = {obj: 1.0 for obj in self.objectives}
        
        composite = 0.0
        total_weight = 0.0
        
        for obj in self.objectives:
            if obj in scores:
                composite += scores[obj] * weights[obj]
                total_weight += weights[obj]
        
        return composite / total_weight if total_weight > 0 else 0.0
    
    def get_multi_objective_info(self) -> Dict[str, Any]:
        """
        Get multi-objective convergence information.
        
        Returns:
            Dictionary with multi-objective convergence information
        """
        info = self.get_convergence_info()
        info["objectives"] = self.objectives
        info["best_scores"] = self.best_scores.copy()
        
        # Calculate convergence for each objective
        objective_convergence = {}
        for obj in self.objectives:
            if len(self.multi_scores[obj]) >= self.window_size:
                recent_scores = self.multi_scores[obj][-self.window_size:]
                variance = np.var(recent_scores)
                objective_convergence[obj] = {
                    "variance": variance,
                    "converged": variance < self.improvement_threshold
                }
        
        info["objective_convergence"] = objective_convergence
        return info
