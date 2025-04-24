"""
TracIn Influence Analysis Module

This module provides functionality for analyzing training sample influence
using the TracIn method.

Main components:
- core/tracin.py: Base TracIn implementation
- core/ranking_tracin.py: Ranking-specific TracIn implementation
- utils/influence_utils.py: Tools for handling influence scores
- utils/visualization.py: Visualization tools
- scripts/: Command-line scripts for influence analysis workflow
"""

from datasets.tracin.core.tracin import TracInCP
from datasets.tracin.core.ranking_tracin import RankingTracInCP
