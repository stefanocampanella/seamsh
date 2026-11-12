# SPDX-FileCopyrightText: 2010-2020 Universite catholique de Louvain (UCL), Belgium
# SPDX-FileCopyrightText: 2026 Stefano Campanella
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Moved out of upstream's seamsh/_tools.py (LGPL, see LICENSE.txt), and rewritten.

"""Progress bars of the long steps (sampling curves, reading features), on the standard error."""

from collections.abc import Iterable
from typing import TypeVar

import tqdm

_T = TypeVar("_T")

_enabled = True


def show_progress(enabled: bool) -> None:
  """Turns the progress bars of seamsh on (the default) or off."""
  global _enabled
  _enabled = bool(enabled)


def progress(iterable: Iterable[_T], desc: str) -> Iterable[_T]:
  """`iterable`, with a progress bar unless they are turned off."""
  return tqdm.tqdm(iterable, desc=desc, disable=not _enabled)
