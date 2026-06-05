"""Built-in games. Importing this package registers them."""

from . import ball  # noqa: F401  (registers "ball")
from . import wheel  # noqa: F401  (registers "wheel")

__all__ = ["ball", "wheel"]
