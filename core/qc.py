from typing import List, Optional
from dataclasses import dataclass
import statistics

@dataclass
class QCResult:
    is_valid: bool
    status: str # OK, UNCERTAIN, OUTLIER, SEVERE
    flags: List[str]

class QualityControl:
    """
    Quality Control (QC) rules for Observations.
    Contract 2.4: Hard Rules + Spatial Rules.
    """
    
    # Thresholds
    TEMP_MIN_C = -50.0
    TEMP_MAX_C = 60.0
    MAD_MULTIPLIER = 3.0
    
    @staticmethod
    def check_hard_rules(temp_c: float, dewpoint_c: Optional[float] = None) -> QCResult:
        """Physical limit checks."""
        flags = []
        
        # Temp limits
        if not (QualityControl.TEMP_MIN_C <= temp_c <= QualityControl.TEMP_MAX_C):
            flags.append("TEMP_OUT_OF_BOUNDS")
            return QCResult(False, "SEVERE", flags)
            
        # Dewpoint valid?
        if dewpoint_c is not None:
             if dewpoint_c > temp_c:
                 flags.append("DEWPOINT_GT_TEMP")
                 # This is physically impossible in standard atmosphere (supersaturation rare at surface)
                 # But sometimes sensors glitch. Mark as UNCERTAIN but usable temp
                 return QCResult(True, "UNCERTAIN", flags)

        return QCResult(True, "OK", flags)

    @staticmethod
    def check_spatial_consistency(
        candidate_val: float, 
        cluster_vals: List[float]
    ) -> QCResult:
        """
        Robust Statistics: Median Absolute Deviation (MAD).
        Checks if candidate is statistical outlier in the cluster.
        """
        if len(cluster_vals) < 3:
            return QCResult(True, "UNCERTAIN", ["LOW_SUPPORT"])
            
        median = statistics.median(cluster_vals)
        deviations = [abs(x - median) for x in cluster_vals]
        mad = statistics.median(deviations)
        
        # Avoid division by zero if all values identical
        if mad == 0:
            mad = 0.1 
            
        # Z-score modified
        score = abs(candidate_val - median) / mad
        
        flags = []
        status = "OK"
        
        if score > QualityControl.MAD_MULTIPLIER:
            flags.append(f"SPATIAL_OUTLIER_MAD_{score:.1f}")
            status = "OUTLIER"
            return QCResult(False, status, flags)
            
        return QCResult(True, status, flags)
