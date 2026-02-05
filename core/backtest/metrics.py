"""
Metrics Module - Prediction and Calibration Metrics

Implements the metrics that matter for HELIOS backtesting:
- Calibration metrics: Brier Score, Log Loss, ECE, Reliability
- Point metrics: MAE, RMSE for tmax_mean
- Stability metrics: Prediction churn, flip count
- T-peak metrics: Bin accuracy, mass near truth

All metrics can be computed per-day or aggregated over periods.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from core.polymarket_labels import normalize_label, parse_label

logger = logging.getLogger("backtest.metrics")


# =============================================================================
# Basic metric functions
# =============================================================================

def brier_score(predicted_prob: float, actual: bool) -> float:
    """
    Brier score for a single prediction.

    Lower is better. Range: [0, 1]

    Args:
        predicted_prob: Predicted probability [0, 1]
        actual: True if event occurred

    Returns:
        Brier score (squared error)
    """
    return (predicted_prob - (1.0 if actual else 0.0)) ** 2


def brier_score_multi(probs: List[float], winner_idx: int) -> float:
    """
    Multi-class Brier score.

    Args:
        probs: List of probabilities for each class (should sum to 1)
        winner_idx: Index of the winning class

    Returns:
        Brier score (mean squared error across classes)
    """
    if not probs or winner_idx < 0 or winner_idx >= len(probs):
        return 1.0

    total = 0.0
    for i, p in enumerate(probs):
        target = 1.0 if i == winner_idx else 0.0
        total += (p - target) ** 2

    return total / len(probs)


def log_loss(predicted_prob: float, actual: bool, eps: float = 1e-15) -> float:
    """
    Log loss (cross-entropy) for a single prediction.

    Lower is better.

    Args:
        predicted_prob: Predicted probability [0, 1]
        actual: True if event occurred
        eps: Small value to avoid log(0)

    Returns:
        Log loss
    """
    p = max(eps, min(1 - eps, predicted_prob))

    if actual:
        return -math.log(p)
    else:
        return -math.log(1 - p)


def log_loss_multi(probs: List[float], winner_idx: int, eps: float = 1e-15) -> float:
    """
    Multi-class log loss.

    Args:
        probs: List of probabilities for each class
        winner_idx: Index of the winning class
        eps: Small value to avoid log(0)

    Returns:
        Log loss
    """
    if not probs or winner_idx < 0 or winner_idx >= len(probs):
        return 10.0  # High penalty

    p = max(eps, min(1 - eps, probs[winner_idx]))
    return -math.log(p)


def expected_calibration_error(
    predictions: List[Tuple[float, bool]],
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE).

    Measures how well predicted probabilities match observed frequencies.

    Args:
        predictions: List of (predicted_prob, actual_outcome) tuples
        n_bins: Number of bins for calibration

    Returns:
        ECE (lower is better)
    """
    if not predictions:
        return 0.0

    # Bin predictions
    bins = [[] for _ in range(n_bins)]

    for prob, actual in predictions:
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bins[bin_idx].append((prob, 1.0 if actual else 0.0))

    # Calculate ECE
    total_samples = len(predictions)
    ece = 0.0

    for bin_data in bins:
        if not bin_data:
            continue

        n = len(bin_data)
        avg_pred = sum(p for p, _ in bin_data) / n
        avg_actual = sum(a for _, a in bin_data) / n

        ece += (n / total_samples) * abs(avg_pred - avg_actual)

    return ece


def reliability_diagram_data(
    predictions: List[Tuple[float, bool]],
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Generate data for reliability diagram.

    Returns:
        Dict with bin_centers, actual_freqs, predicted_means, counts
    """
    if not predictions:
        return {
            "bin_centers": [],
            "actual_freqs": [],
            "predicted_means": [],
            "counts": []
        }

    bins = [[] for _ in range(n_bins)]

    for prob, actual in predictions:
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bins[bin_idx].append((prob, 1.0 if actual else 0.0))

    bin_centers = []
    actual_freqs = []
    predicted_means = []
    counts = []

    for i, bin_data in enumerate(bins):
        center = (i + 0.5) / n_bins
        bin_centers.append(center)

        if bin_data:
            predicted_means.append(sum(p for p, _ in bin_data) / len(bin_data))
            actual_freqs.append(sum(a for _, a in bin_data) / len(bin_data))
            counts.append(len(bin_data))
        else:
            predicted_means.append(center)
            actual_freqs.append(None)
            counts.append(0)

    return {
        "bin_centers": bin_centers,
        "actual_freqs": actual_freqs,
        "predicted_means": predicted_means,
        "counts": counts
    }


def sharpness(probs_list: List[List[float]]) -> float:
    """
    Measure sharpness of probability distributions.

    Higher is better (more confident predictions).
    Uses entropy: lower entropy = sharper.

    Returns:
        Average "sharpness" (1 - normalized entropy)
    """
    if not probs_list:
        return 0.0

    total_sharpness = 0.0

    for probs in probs_list:
        if not probs:
            continue

        # Calculate entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p)

        # Normalize by max entropy (uniform distribution)
        max_entropy = math.log(len(probs))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
            total_sharpness += 1 - normalized_entropy

    return total_sharpness / len(probs_list) if probs_list else 0.0


# =============================================================================
# Point prediction metrics
# =============================================================================

def mae(predictions: List[Tuple[float, float]]) -> float:
    """
    Mean Absolute Error.

    Args:
        predictions: List of (predicted, actual) tuples

    Returns:
        MAE
    """
    if not predictions:
        return 0.0

    return sum(abs(p - a) for p, a in predictions) / len(predictions)


def rmse(predictions: List[Tuple[float, float]]) -> float:
    """
    Root Mean Squared Error.

    Args:
        predictions: List of (predicted, actual) tuples

    Returns:
        RMSE
    """
    if not predictions:
        return 0.0

    mse = sum((p - a) ** 2 for p, a in predictions) / len(predictions)
    return math.sqrt(mse)


def bias(predictions: List[Tuple[float, float]]) -> float:
    """
    Mean bias (systematic over/under prediction).

    Positive = over-predicting, Negative = under-predicting

    Args:
        predictions: List of (predicted, actual) tuples

    Returns:
        Mean bias
    """
    if not predictions:
        return 0.0

    return sum(p - a for p, a in predictions) / len(predictions)


# =============================================================================
# Stability metrics
# =============================================================================

def prediction_churn(
    nowcast_sequence: List[Dict],
    bucket_key: str = "p_bucket"
) -> float:
    """
    Measure prediction churn (instability) over time.

    Higher = more unstable predictions.

    Args:
        nowcast_sequence: List of nowcast outputs in time order
        bucket_key: Key for bucket probabilities

    Returns:
        Sum of absolute probability changes
    """
    if len(nowcast_sequence) < 2:
        return 0.0

    total_churn = 0.0

    for i in range(1, len(nowcast_sequence)):
        prev = nowcast_sequence[i - 1].get(bucket_key, [])
        curr = nowcast_sequence[i].get(bucket_key, [])

        if not prev or not curr:
            continue

        # Handle both list of dicts and list of floats
        if isinstance(prev[0], dict):
            prev_probs = [b.get("probability", b.get("prob", 0)) for b in prev]
            curr_probs = [b.get("probability", b.get("prob", 0)) for b in curr]
        else:
            prev_probs = prev
            curr_probs = curr

        # Calculate change
        for p1, p2 in zip(prev_probs, curr_probs):
            total_churn += abs(p2 - p1)

    return total_churn


def flip_count(
    nowcast_sequence: List[Dict],
    bucket_key: str = "p_bucket"
) -> int:
    """
    Count how many times the predicted winner changes.

    Args:
        nowcast_sequence: List of nowcast outputs in time order
        bucket_key: Key for bucket probabilities

    Returns:
        Number of flips
    """
    if len(nowcast_sequence) < 2:
        return 0

    def get_winner_idx(nowcast: Dict) -> int:
        buckets = nowcast.get(bucket_key, [])
        if not buckets:
            return -1

        if isinstance(buckets[0], dict):
            probs = [b.get("probability", b.get("prob", 0)) for b in buckets]
        else:
            probs = buckets

        return probs.index(max(probs)) if probs else -1

    flips = 0
    prev_winner = get_winner_idx(nowcast_sequence[0])

    for nowcast in nowcast_sequence[1:]:
        curr_winner = get_winner_idx(nowcast)
        if curr_winner != prev_winner and curr_winner >= 0:
            flips += 1
            prev_winner = curr_winner

    return flips


# =============================================================================
# T-peak metrics
# =============================================================================

def tpeak_accuracy(
    predictions: List[Tuple[str, str]]
) -> float:
    """
    Accuracy of t-peak bin predictions.

    Args:
        predictions: List of (predicted_bin, actual_bin) tuples

    Returns:
        Accuracy (0-1)
    """
    if not predictions:
        return 0.0

    correct = sum(1 for p, a in predictions if p == a)
    return correct / len(predictions)


def tpeak_mass_near_truth(
    nowcast: Dict,
    actual_bin: str,
    tolerance: int = 1
) -> float:
    """
    Probability mass within tolerance of the true bin.

    Args:
        nowcast: Nowcast output with t_peak_bins
        actual_bin: The actual t-peak bin (e.g., "14-16")
        tolerance: Number of adjacent bins to include

    Returns:
        Probability mass near truth
    """
    bins = nowcast.get("t_peak_bins", [])
    if not bins:
        return 0.0

    # Find actual bin index
    actual_idx = None
    for i, b in enumerate(bins):
        label = b.get("label_short", "") or b.get("label", "")
        if label == actual_bin or actual_bin in label:
            actual_idx = i
            break

    if actual_idx is None:
        return 0.0

    # Sum probability mass in range
    total_mass = 0.0
    for i, b in enumerate(bins):
        if abs(i - actual_idx) <= tolerance:
            total_mass += b.get("probability", 0)

    return total_mass


# =============================================================================
# Calibration Metrics Aggregator
# =============================================================================

@dataclass
class CalibrationMetrics:
    """Aggregated calibration metrics for a period."""

    # Sample info
    n_samples: int = 0
    n_days: int = 0

    # Brier scores
    brier_global: float = 0.0
    brier_by_bucket: Dict[str, float] = field(default_factory=dict)

    # Log loss
    log_loss_global: float = 0.0

    # ECE
    ece: float = 0.0

    # Reliability diagram data
    reliability_data: Dict[str, Any] = field(default_factory=dict)

    # Sharpness
    sharpness: float = 0.0

    # Point metrics (for tmax_mean)
    tmax_mae: float = 0.0
    tmax_rmse: float = 0.0
    tmax_bias: float = 0.0

    # Stability metrics
    avg_churn: float = 0.0
    avg_flips: float = 0.0

    # T-peak metrics
    tpeak_accuracy: float = 0.0
    tpeak_mass_near: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_days": self.n_days,
            "calibration": {
                "brier_global": self.brier_global,
                "brier_by_bucket": self.brier_by_bucket,
                "log_loss_global": self.log_loss_global,
                "ece": self.ece,
                "sharpness": self.sharpness,
            },
            "point_metrics": {
                "tmax_mae": self.tmax_mae,
                "tmax_rmse": self.tmax_rmse,
                "tmax_bias": self.tmax_bias,
            },
            "stability": {
                "avg_churn": self.avg_churn,
                "avg_flips": self.avg_flips,
            },
            "tpeak": {
                "accuracy": self.tpeak_accuracy,
                "mass_near_truth": self.tpeak_mass_near,
            },
            "reliability_data": self.reliability_data,
        }

    def summary(self) -> str:
        """Generate text summary."""
        return f"""
Calibration Metrics Summary
===========================
Samples: {self.n_samples} predictions over {self.n_days} days

Calibration:
  Brier Score: {self.brier_global:.4f}
  Log Loss: {self.log_loss_global:.4f}
  ECE: {self.ece:.4f}
  Sharpness: {self.sharpness:.4f}

Point Predictions (Tmax):
  MAE: {self.tmax_mae:.2f}°F
  RMSE: {self.tmax_rmse:.2f}°F
  Bias: {self.tmax_bias:+.2f}°F

Stability:
  Avg Churn: {self.avg_churn:.4f}
  Avg Flips: {self.avg_flips:.1f}

T-Peak:
  Accuracy: {self.tpeak_accuracy:.2%}
  Mass Near Truth: {self.tpeak_mass_near:.2%}
"""


class MetricsCalculator:
    """
    Calculates and aggregates all backtest metrics.
    """

    def __init__(self):
        # Accumulators
        self._bucket_predictions: List[Tuple[List[float], int]] = []  # (probs, winner_idx)
        self._tmax_predictions: List[Tuple[float, float]] = []  # (predicted, actual)
        self._tpeak_predictions: List[Tuple[str, str]] = []  # (predicted, actual)
        self._churn_values: List[float] = []
        self._flip_counts: List[int] = []
        self._probs_for_sharpness: List[List[float]] = []
        self._days_processed: int = 0

    def reset(self):
        """Reset all accumulators."""
        self._bucket_predictions = []
        self._tmax_predictions = []
        self._tpeak_predictions = []
        self._churn_values = []
        self._flip_counts = []
        self._probs_for_sharpness = []
        self._days_processed = 0

    def add_day(
        self,
        nowcast_sequence: List[Dict],
        label: "DayLabel",
        final_nowcast: Optional[Dict] = None
    ):
        """
        Add a day's worth of predictions for metric calculation.

        Args:
            nowcast_sequence: All nowcast outputs for the day (time-ordered)
            label: Ground truth label for the day
            final_nowcast: Optional final nowcast to use for bucket evaluation
                          (defaults to last in sequence)
        """
        if not nowcast_sequence:
            return

        self._days_processed += 1

        # Use final nowcast for bucket/tmax evaluation
        if final_nowcast is None:
            final_nowcast = nowcast_sequence[-1]

        # Extract bucket probabilities
        p_bucket = final_nowcast.get("p_bucket", [])
        if p_bucket:
            if isinstance(p_bucket[0], dict):
                # Support both "prob" and "probability" keys
                probs = [b.get("prob", b.get("probability", 0)) for b in p_bucket]
                labels = [normalize_label(b.get("label", "")) for b in p_bucket]
            else:
                probs = p_bucket
                labels = []

            # Normalize probabilities if they don't sum to ~1
            prob_sum = sum(probs)
            if prob_sum > 0 and abs(prob_sum - 1.0) > 0.01:
                probs = [p / prob_sum for p in probs]

            # Find winner index by matching label or temperature
            winner_idx = -1
            label_bucket = normalize_label(label.y_bucket_winner) if label.y_bucket_winner else None
            if label_bucket and labels:
                # Try exact label match first
                for i, lbl in enumerate(labels):
                    if lbl == label_bucket:
                        winner_idx = i
                        break

                # If no exact match, find bucket by temperature
                if winner_idx < 0 and label.y_tmax_aligned is not None:
                    actual_temp = label.y_tmax_aligned
                    for i, lbl in enumerate(labels):
                        kind, low, high = parse_label(lbl)
                        if kind == "below" and high is not None and actual_temp <= high:
                            winner_idx = i
                            break
                        if kind == "above" and low is not None and actual_temp >= low:
                            winner_idx = i
                            break
                        if kind == "range" and low is not None and high is not None:
                            if low <= actual_temp <= high:
                                winner_idx = i
                                break
                        if kind == "single" and low is not None and actual_temp == low:
                            winner_idx = i
                            break

                    # If still no match, use edge bucket (below lowest or above highest)
                    if winner_idx < 0 and labels:
                        # Temperature is outside the range - assign to nearest edge
                        first_kind, first_low, first_high = parse_label(labels[0])
                        last_kind, last_low, last_high = parse_label(labels[-1])
                        if first_kind == "below" and first_high is not None and actual_temp <= first_high:
                            winner_idx = 0
                        elif first_low is not None and actual_temp < first_low:
                            winner_idx = 0
                        elif last_kind == "above" and last_low is not None and actual_temp >= last_low:
                            winner_idx = len(labels) - 1
                        elif last_high is not None and actual_temp >= last_high:
                            winner_idx = len(labels) - 1

            elif label.y_bucket_index is not None:
                winner_idx = label.y_bucket_index

            if winner_idx >= 0 and probs:
                self._bucket_predictions.append((probs, winner_idx))
                self._probs_for_sharpness.append(probs)

        # Tmax prediction
        tmax_pred = final_nowcast.get("tmax_mean_f") or final_nowcast.get("tmax_mean")
        if tmax_pred is not None and label.y_tmax_aligned is not None:
            self._tmax_predictions.append((tmax_pred, label.y_tmax_aligned))

        # T-peak prediction (use highest probability bin)
        tpeak_bins = final_nowcast.get("t_peak_bins", [])
        if tpeak_bins and label.y_t_peak_bin:
            # Find predicted peak bin
            max_prob = 0
            pred_bin = None
            for b in tpeak_bins:
                # Support both "prob" and "probability" keys
                prob = b.get("prob", b.get("probability", 0))
                if prob > max_prob:
                    max_prob = prob
                    pred_bin = b.get("label_short", b.get("label", ""))

            if pred_bin:
                self._tpeak_predictions.append((pred_bin, label.y_t_peak_bin))

        # Stability metrics
        churn = prediction_churn(nowcast_sequence)
        flips = flip_count(nowcast_sequence)
        self._churn_values.append(churn)
        self._flip_counts.append(flips)

    def compute(self) -> CalibrationMetrics:
        """
        Compute all metrics from accumulated data.

        Returns:
            CalibrationMetrics with all computed values
        """
        metrics = CalibrationMetrics()
        metrics.n_days = self._days_processed
        metrics.n_samples = len(self._bucket_predictions)

        # Brier score
        if self._bucket_predictions:
            brier_values = [
                brier_score_multi(probs, winner)
                for probs, winner in self._bucket_predictions
                if winner >= 0
            ]
            if brier_values:
                metrics.brier_global = sum(brier_values) / len(brier_values)

            # Log loss
            ll_values = [
                log_loss_multi(probs, winner)
                for probs, winner in self._bucket_predictions
                if winner >= 0
            ]
            if ll_values:
                metrics.log_loss_global = sum(ll_values) / len(ll_values)

        # ECE (for highest probability bucket)
        binary_predictions = []
        for probs, winner in self._bucket_predictions:
            if winner >= 0 and probs:
                max_idx = probs.index(max(probs))
                binary_predictions.append((max(probs), max_idx == winner))

        if binary_predictions:
            metrics.ece = expected_calibration_error(binary_predictions)
            metrics.reliability_data = reliability_diagram_data(binary_predictions)

        # Sharpness
        metrics.sharpness = sharpness(self._probs_for_sharpness)

        # Point metrics
        if self._tmax_predictions:
            metrics.tmax_mae = mae(self._tmax_predictions)
            metrics.tmax_rmse = rmse(self._tmax_predictions)
            metrics.tmax_bias = bias(self._tmax_predictions)

        # Stability
        if self._churn_values:
            metrics.avg_churn = sum(self._churn_values) / len(self._churn_values)
        if self._flip_counts:
            metrics.avg_flips = sum(self._flip_counts) / len(self._flip_counts)

        # T-peak
        if self._tpeak_predictions:
            metrics.tpeak_accuracy = tpeak_accuracy(self._tpeak_predictions)

        return metrics

    def compute_for_day(
        self,
        nowcast_sequence: List[Dict],
        label: "DayLabel"
    ) -> CalibrationMetrics:
        """
        Compute metrics for a single day (without affecting accumulators).
        """
        calc = MetricsCalculator()
        calc.add_day(nowcast_sequence, label)
        return calc.compute()


# =============================================================================
# Segmentation helpers
# =============================================================================

def segment_by_regime(
    days_data: List[Tuple[List[Dict], "DayLabel", Dict]],
    regime_key: str = "regime"
) -> Dict[str, CalibrationMetrics]:
    """
    Compute metrics segmented by regime.

    Args:
        days_data: List of (nowcast_sequence, label, metadata) tuples
        regime_key: Key in metadata for regime classification

    Returns:
        Dict mapping regime names to metrics
    """
    by_regime = defaultdict(list)

    for nowcast_seq, label, metadata in days_data:
        regime = metadata.get(regime_key, "unknown")
        by_regime[regime].append((nowcast_seq, label))

    results = {}
    for regime, day_list in by_regime.items():
        calc = MetricsCalculator()
        for nowcast_seq, label in day_list:
            calc.add_day(nowcast_seq, label)
        results[regime] = calc.compute()

    return results
