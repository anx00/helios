"""
Opportunity Detection Module

Identifies betting opportunities by comparing predictions with markets.
"""
from .detector import (
    BettingOpportunity,
    check_bet_opportunity,
    format_opportunity_display
)

__all__ = [
    'BettingOpportunity',
    'check_bet_opportunity',
    'format_opportunity_display'
]
