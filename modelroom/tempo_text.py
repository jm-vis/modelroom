"""The sentences of the speed, design and memory lines of AP9-T, in one place.

AP9-T-L put the first one here: the line that names a website for model quality. T-G, T-M and T-K
add their sentences next to it, so a wording change is a change to this file and its tests, not a
hunt through the views. Text only -- nothing here reads a document or opens an address.
"""

from __future__ import annotations

# One line under the result of step 5, and after `modelroom render`: where a reader can see how good
# these models are at their own task. modelroom never opens the address (README, "Network access").
QUALITY_LINK_LINE = "Quality of these models for your task: https://artificialanalysis.ai"

__all__ = ["QUALITY_LINK_LINE"]
