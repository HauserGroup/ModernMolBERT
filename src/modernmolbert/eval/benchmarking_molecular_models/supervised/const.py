import os

CV_SPLITS = 5
# Production --missing-labels as-negative scoring passes cv=5 to GridSearchCV.
# Scikit-learn uses unshuffled StratifiedKFold for binary classification and
# unshuffled KFold for multi-output classification in that path.
PRODUCTION_CV_POLICY = {"folds": CV_SPLITS, "shuffle": False, "seed": None}
# Use actual core count so division by memory_weight stays meaningful; 32 caused
# context-switch thrash on Mac where core counts are typically 8–16.
N_JOBS = os.cpu_count() or 1
DEFAULT_MEMORY_WEIGHT = 1  # Actual N_JOBS will be N_JOBS / memory_weight
VERBOSITY = 1
