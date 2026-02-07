"""
Probability Calibrator Module

Calibrates nowcast P(bucket) probabilities using historical data.
Two methods are provided:

1. Temperature Scaling: single scalar T that minimizes NLL on logits.
   T > 1 means overconfident, T < 1 means underconfident.

2. Isotonic Regression (PAV): non-parametric monotonic mapping
   from raw probability to calibrated probability.

Both methods use PURE PYTHON fallbacks when scipy/sklearn are unavailable.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("backtest.prob_calibrator")

EPS = 1e-15


# =============================================================================
# Pure-Python math helpers
# =============================================================================

def _clamp(x: float, lo: float = EPS, hi: float = 1.0 - EPS) -> float:
    return max(lo, min(hi, x))


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _softmax(logits: List[float]) -> List[float]:
    max_l = max(logits) if logits else 0.0
    exps = [math.exp(l - max_l) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _nll(probs_list: List[List[float]], labels: List[int]) -> float:
    """Negative log-likelihood across samples."""
    total = 0.0
    for probs, winner_idx in zip(probs_list, labels):
        if 0 <= winner_idx < len(probs):
            total -= math.log(_clamp(probs[winner_idx]))
    return total / max(len(probs_list), 1)


# =============================================================================
# Pool Adjacent Violators (PAV) -- pure Python isotonic regression
# =============================================================================

def _pav(values: List[float], weights: Optional[List[float]] = None) -> List[float]:
    """
    Pool Adjacent Violators algorithm for non-decreasing isotonic regression.

    Given a sequence of values, returns the nearest non-decreasing sequence
    that minimizes weighted squared error.

    Args:
        values: raw values to make monotone
        weights: optional sample weights (default: uniform)

    Returns:
        List of isotonically-regressed values (same length as input)
    """
    n = len(values)
    if n == 0:
        return []
    if weights is None:
        weights = [1.0] * n

    # Each block is [sum_of_weighted_values, sum_of_weights, start_idx, end_idx]
    blocks: List[List[float]] = []
    for i in range(n):
        blocks.append([values[i] * weights[i], weights[i], i, i])
        # Merge with previous block while violating monotonicity
        while len(blocks) >= 2:
            curr = blocks[-1]
            prev = blocks[-2]
            prev_mean = prev[0] / prev[1] if prev[1] > 0 else 0
            curr_mean = curr[0] / curr[1] if curr[1] > 0 else 0
            if prev_mean > curr_mean:
                # Merge: pool the two blocks
                prev[0] += curr[0]
                prev[1] += curr[1]
                prev[3] = curr[3]  # extend end_idx
                blocks.pop()
            else:
                break

    # Expand blocks back to full-length output
    result = [0.0] * n
    for wv, w, start, end in blocks:
        val = wv / w if w > 0 else 0.0
        for i in range(int(start), int(end) + 1):
            result[i] = val

    return result


# =============================================================================
# ProbabilityCalibrator
# =============================================================================

@dataclass
class IsotonicBreakpoints:
    """Stores fitted isotonic regression breakpoints for serialization."""
    x_points: List[float] = field(default_factory=list)
    y_points: List[float] = field(default_factory=list)


@dataclass
class CalibrationParams:
    """Serializable calibration parameters."""
    method: str = "temperature_scaling"
    temperature: float = 1.0
    isotonic: Optional[IsotonicBreakpoints] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "method": self.method,
            "temperature": self.temperature,
        }
        if self.isotonic:
            d["isotonic"] = {
                "x_points": self.isotonic.x_points,
                "y_points": self.isotonic.y_points,
            }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationParams":
        params = cls(
            method=d.get("method", "temperature_scaling"),
            temperature=d.get("temperature", 1.0),
        )
        iso = d.get("isotonic")
        if iso:
            params.isotonic = IsotonicBreakpoints(
                x_points=iso.get("x_points", []),
                y_points=iso.get("y_points", []),
            )
        return params


class ProbabilityCalibrator:
    """
    Calibrates nowcast P(bucket) probabilities using historical data.

    Methods:
      - "temperature_scaling": single scalar T applied to logits
      - "isotonic": non-parametric monotonic mapping via PAV

    Usage:
        cal = ProbabilityCalibrator(method="temperature_scaling")
        cal.fit(predictions, labels)
        calibrated = cal.calibrate(raw_p_bucket)
    """

    def __init__(self, method: str = "temperature_scaling"):
        if method not in ("temperature_scaling", "isotonic"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.temperature: float = 1.0
        self._isotonic_x: List[float] = []
        self._isotonic_y: List[float] = []
        self._fitted = False

        # Raw data kept for reliability diagram
        self._raw_pairs: List[Tuple[float, float]] = []  # (predicted, actual_indicator)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        predictions: List[Dict],
        labels: List["DayLabel"],
    ) -> None:
        """
        Fit the calibrator on training data.

        Args:
            predictions: list of nowcast dicts, each containing "p_bucket"
                         (list of dicts with "label" and "prob"/"probability")
            labels: list of DayLabel with y_bucket_winner
        """
        if len(predictions) != len(labels):
            raise ValueError(
                f"predictions ({len(predictions)}) and labels ({len(labels)}) "
                f"must have the same length"
            )

        # Extract (probs_vector, winner_idx) pairs
        probs_list: List[List[float]] = []
        winner_indices: List[int] = []
        binary_pairs: List[Tuple[float, float]] = []  # for isotonic

        from core.polymarket_labels import normalize_label

        for pred, lbl in zip(predictions, labels):
            p_bucket = pred.get("p_bucket", [])
            if not p_bucket:
                continue

            if isinstance(p_bucket[0], dict):
                probs = [b.get("prob", b.get("probability", 0.0)) for b in p_bucket]
                bucket_labels = [normalize_label(b.get("label", "")) for b in p_bucket]
            else:
                probs = list(p_bucket)
                bucket_labels = []

            if not probs or not lbl.y_bucket_winner:
                continue

            # Normalize probs to sum to 1
            psum = sum(probs)
            if psum <= 0:
                continue
            probs = [p / psum for p in probs]

            # Find winner index
            target_label = normalize_label(lbl.y_bucket_winner)
            winner_idx = -1
            if bucket_labels:
                for i, bl in enumerate(bucket_labels):
                    if bl == target_label:
                        winner_idx = i
                        break

            if winner_idx < 0:
                if lbl.y_bucket_index is not None and 0 <= lbl.y_bucket_index < len(probs):
                    winner_idx = lbl.y_bucket_index
                else:
                    continue

            probs_list.append(probs)
            winner_indices.append(winner_idx)

            # Build binary pairs: for each bucket, record (pred_prob, 1 if winner else 0)
            for i, p in enumerate(probs):
                indicator = 1.0 if i == winner_idx else 0.0
                binary_pairs.append((p, indicator))

        if not probs_list:
            logger.warning("No valid (prediction, label) pairs for calibration fit")
            return

        self._raw_pairs = binary_pairs

        if self.method == "temperature_scaling":
            self._fit_temperature(probs_list, winner_indices)
        elif self.method == "isotonic":
            self._fit_isotonic(binary_pairs)

        self._fitted = True
        logger.info(
            "Fitted ProbabilityCalibrator (method=%s) on %d samples",
            self.method, len(probs_list),
        )

    def _fit_temperature(
        self,
        probs_list: List[List[float]],
        winner_indices: List[int],
    ) -> None:
        """
        Find temperature T that minimizes NLL: calibrated_p = softmax(logits / T).
        """
        # Convert probs to logits
        logits_list = []
        for probs in probs_list:
            logits_list.append([_logit(p) for p in probs])

        def nll_at_t(t: float) -> float:
            total = 0.0
            for logits, winner in zip(logits_list, winner_indices):
                scaled = [l / t for l in logits]
                cal_probs = _softmax(scaled)
                total -= math.log(_clamp(cal_probs[winner]))
            return total / len(logits_list)

        # Try scipy first for efficient optimization
        best_t = 1.0
        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(nll_at_t, bounds=(0.1, 5.0), method="bounded")
            best_t = res.x
            logger.debug("Temperature scaling via scipy: T=%.4f, NLL=%.6f", best_t, res.fun)
        except ImportError:
            # Fallback: grid search over T in [0.1, 5.0], step 0.05
            best_nll = float("inf")
            t = 0.1
            while t <= 5.0:
                val = nll_at_t(t)
                if val < best_nll:
                    best_nll = val
                    best_t = t
                t += 0.05
            # Refine around best_t with finer step
            refine_lo = max(0.1, best_t - 0.1)
            refine_hi = min(5.0, best_t + 0.1)
            t = refine_lo
            while t <= refine_hi:
                val = nll_at_t(t)
                if val < best_nll:
                    best_nll = val
                    best_t = t
                t += 0.005
            logger.debug("Temperature scaling via grid: T=%.4f, NLL=%.6f", best_t, best_nll)

        self.temperature = best_t

    def _fit_isotonic(self, binary_pairs: List[Tuple[float, float]]) -> None:
        """
        Fit isotonic regression on binary (predicted_prob, actual_indicator) pairs.

        Sorts by predicted prob, runs PAV, then stores breakpoints for interpolation.
        """
        if not binary_pairs:
            return

        # Sort by predicted probability
        sorted_pairs = sorted(binary_pairs, key=lambda x: x[0])

        preds = [p for p, _ in sorted_pairs]
        actuals = [a for _, a in sorted_pairs]

        # Run PAV
        calibrated = _pav(actuals)

        # Store as piecewise-linear breakpoints (deduplicate flat regions)
        x_points = []
        y_points = []
        prev_y = None
        for x, y in zip(preds, calibrated):
            if prev_y is None or abs(y - prev_y) > 1e-9 or x == preds[-1]:
                x_points.append(x)
                y_points.append(y)
                prev_y = y

        # Ensure endpoints
        if x_points and x_points[0] > 0.0:
            x_points.insert(0, 0.0)
            y_points.insert(0, y_points[0])
        if x_points and x_points[-1] < 1.0:
            x_points.append(1.0)
            y_points.append(y_points[-1])

        self._isotonic_x = x_points
        self._isotonic_y = y_points

    # ------------------------------------------------------------------
    # calibrate
    # ------------------------------------------------------------------

    def calibrate(self, p_bucket: List[Dict]) -> List[Dict]:
        """
        Apply fitted calibration to raw p_bucket.

        Args:
            p_bucket: list of dicts with "label" and "prob"/"probability"

        Returns:
            New list of dicts with calibrated probabilities
        """
        if not self._fitted or not p_bucket:
            return p_bucket

        if self.method == "temperature_scaling":
            return self._calibrate_temperature(p_bucket)
        elif self.method == "isotonic":
            return self._calibrate_isotonic(p_bucket)
        return p_bucket

    def _calibrate_temperature(self, p_bucket: List[Dict]) -> List[Dict]:
        """Apply temperature scaling: softmax(logits / T)."""
        # Extract probs
        if isinstance(p_bucket[0], dict):
            probs = [b.get("prob", b.get("probability", 0.0)) for b in p_bucket]
        else:
            probs = list(p_bucket)

        psum = sum(probs)
        if psum <= 0:
            return p_bucket
        probs = [p / psum for p in probs]

        # Convert to logits, scale, softmax
        logits = [_logit(p) for p in probs]
        scaled = [l / self.temperature for l in logits]
        cal_probs = _softmax(scaled)

        # Rebuild output
        result = []
        for i, b in enumerate(p_bucket):
            if isinstance(b, dict):
                new_b = dict(b)
                if "prob" in new_b:
                    new_b["prob"] = cal_probs[i]
                if "probability" in new_b:
                    new_b["probability"] = cal_probs[i]
                # If neither key was present, add "prob"
                if "prob" not in new_b and "probability" not in new_b:
                    new_b["prob"] = cal_probs[i]
                result.append(new_b)
            else:
                result.append(cal_probs[i])

        return result

    def _calibrate_isotonic(self, p_bucket: List[Dict]) -> List[Dict]:
        """Apply isotonic regression mapping to each bucket probability."""
        if not self._isotonic_x:
            return p_bucket

        if isinstance(p_bucket[0], dict):
            probs = [b.get("prob", b.get("probability", 0.0)) for b in p_bucket]
        else:
            probs = list(p_bucket)

        psum = sum(probs)
        if psum <= 0:
            return p_bucket
        probs = [p / psum for p in probs]

        # Map each prob through isotonic breakpoints
        cal_probs = [self._isotonic_lookup(p) for p in probs]

        # Re-normalize to sum to 1
        csum = sum(cal_probs)
        if csum > 0:
            cal_probs = [p / csum for p in cal_probs]

        result = []
        for i, b in enumerate(p_bucket):
            if isinstance(b, dict):
                new_b = dict(b)
                if "prob" in new_b:
                    new_b["prob"] = cal_probs[i]
                if "probability" in new_b:
                    new_b["probability"] = cal_probs[i]
                if "prob" not in new_b and "probability" not in new_b:
                    new_b["prob"] = cal_probs[i]
                result.append(new_b)
            else:
                result.append(cal_probs[i])

        return result

    def _isotonic_lookup(self, x: float) -> float:
        """Piecewise-linear interpolation on isotonic breakpoints."""
        xs = self._isotonic_x
        ys = self._isotonic_y

        if not xs:
            return x

        # Clamp
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]

        # Binary search for interval
        lo, hi = 0, len(xs) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid

        # Linear interpolation
        x0, x1 = xs[lo], xs[hi]
        y0, y1 = ys[lo], ys[hi]
        if abs(x1 - x0) < EPS:
            return y0

        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    # ------------------------------------------------------------------
    # reliability data
    # ------------------------------------------------------------------

    def get_reliability_data(self, n_bins: int = 10) -> List[Tuple[float, float, float, int]]:
        """
        Return binned reliability diagram data from the training fit.

        Returns:
            List of (bin_center, predicted_avg, actual_freq, count) tuples.
            Only bins with count > 0 are included.
        """
        if not self._raw_pairs:
            return []

        bins: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
        for pred, actual in self._raw_pairs:
            idx = min(int(pred * n_bins), n_bins - 1)
            bins[idx].append((pred, actual))

        result = []
        for i, bin_data in enumerate(bins):
            center = (i + 0.5) / n_bins
            if bin_data:
                pred_avg = sum(p for p, _ in bin_data) / len(bin_data)
                actual_freq = sum(a for _, a in bin_data) / len(bin_data)
                result.append((center, pred_avg, actual_freq, len(bin_data)))
            else:
                result.append((center, center, 0.0, 0))

        return result

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def get_params(self) -> CalibrationParams:
        """Get serializable calibration parameters."""
        params = CalibrationParams(
            method=self.method,
            temperature=self.temperature,
        )
        if self.method == "isotonic" and self._isotonic_x:
            params.isotonic = IsotonicBreakpoints(
                x_points=list(self._isotonic_x),
                y_points=list(self._isotonic_y),
            )
        return params

    def load_params(self, params: CalibrationParams) -> None:
        """Load previously fitted calibration parameters."""
        self.method = params.method
        self.temperature = params.temperature
        if params.isotonic:
            self._isotonic_x = list(params.isotonic.x_points)
            self._isotonic_y = list(params.isotonic.y_points)
        self._fitted = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return self.get_params().to_dict()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProbabilityCalibrator":
        """Deserialize from dict."""
        cal = cls(method=d.get("method", "temperature_scaling"))
        params = CalibrationParams.from_dict(d)
        cal.load_params(params)
        return cal
