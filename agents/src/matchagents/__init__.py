"""The AI broadcast desk: seven agents that explain a match, with every number verified.

Every model call goes through the budget guard, and nothing reaches the screen until the
fact-checker has approved it.
"""

from .budget import BudgetExceeded, BudgetGuard

__all__ = ["BudgetExceeded", "BudgetGuard"]
