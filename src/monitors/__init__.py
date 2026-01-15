"""Monitoring and autorater implementations for LORE.

This module provides:
- LindnerAutorater: CoT obfuscation detection based on Lindner et al. methodology
- EnsembleMonitor: Multi-judge ensemble for robust obfuscation detection
- ObfuscationDetector: General obfuscation detection methods
- Utility functions for Regime B fitness calculation
"""

from src.monitors.lindner_autorater import (
    # Main class
    LindnerAutorater,
    # Result types
    AutoraterResult,
    PatternMatch,
    # Enums
    ObfuscationPattern,
    RiskLevel,
    # Rubrics
    RUBRIC_V1,
    RUBRIC_V2,
    # Utility functions
    calculate_regime_b_fitness,
    calculate_obfuscation_score,
    create_autorater,
)
from src.monitors.ensemble import (
    # Main class
    EnsembleMonitor,
    # Result types
    EnsembleResult,
    JudgeRating,
    CalibrationResult,
    # Enums
    AgreementLevel,
    CalibrationStatus,
    # Calibration data
    BENIGN_COT_TEMPLATES,
    OBFUSCATED_COT_TEMPLATES,
    generate_calibration_cots,
    # Utility functions
    create_ensemble_monitor,
    calculate_ensemble_regime_b_fitness,
)
from src.monitors.detection import (
    ObfuscationDetector,
    DetectionResult,
    ParaphraseConsistencyChecker,
    AttentionAnalyzer,
)

__all__ = [
    # Lindner Autorater
    "LindnerAutorater",
    "AutoraterResult",
    "PatternMatch",
    "ObfuscationPattern",
    "RiskLevel",
    "RUBRIC_V1",
    "RUBRIC_V2",
    "calculate_regime_b_fitness",
    "calculate_obfuscation_score",
    "create_autorater",
    # Ensemble Monitor
    "EnsembleMonitor",
    "EnsembleResult",
    "JudgeRating",
    "CalibrationResult",
    "AgreementLevel",
    "CalibrationStatus",
    "BENIGN_COT_TEMPLATES",
    "OBFUSCATED_COT_TEMPLATES",
    "generate_calibration_cots",
    "create_ensemble_monitor",
    "calculate_ensemble_regime_b_fitness",
    # Detection
    "ObfuscationDetector",
    "DetectionResult",
    "ParaphraseConsistencyChecker",
    "AttentionAnalyzer",
]
