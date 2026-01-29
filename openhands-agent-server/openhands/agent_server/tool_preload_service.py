"""Service which preloads chromium."""

from __future__ import annotations

import sys

from openhands.agent_server.config import get_default_config
from openhands.sdk.logger import get_logger
from openhands.sdk.tool.schema import Action
from openhands.sdk.tool.tool import create_action_type_with_risk
from openhands.sdk.utils.models import get_known_concrete_subclasses


_logger = get_logger(__name__)


class ToolPreloadService:
    """Service which preloads tools / chromium reducing time to
    start first conversation"""

    running: bool = False

    async def start(self) -> bool:
        """Preload tools"""

        # Skip if already running
        if self.running:
            return True

        self.running = True
        
        # Pre-creating all *WithRisk classes prevents processing which costs
        # significant time per tool on the first conversation invocation.
        # This MUST run even if browser preload fails, because model_rebuild
        # needs these classes to exist when parsing Action union types.
        try:
            for action_type in get_known_concrete_subclasses(Action):
                create_action_type_with_risk(action_type)
            _logger.debug("Pre-created all Action WithRisk classes")
        except Exception:
            _logger.exception("Error creating WithRisk classes")
            return False
        
        # Try to preload browser (optional, can fail without affecting other tools)
        try:
            if sys.platform == "win32":
                from openhands.tools.browser_use.impl_windows import (
                    WindowsBrowserToolExecutor as BrowserToolExecutor,
                )
            else:
                from openhands.tools.browser_use.impl import BrowserToolExecutor

            # Creating an instance here to preload chomium
            BrowserToolExecutor()
            _logger.debug(f"Loaded {BrowserToolExecutor}")
        except Exception:
            _logger.warning("Browser preload failed (this is optional and won't affect other tools)")
            
        return True

    async def stop(self) -> None:
        """Stop the tool preload process."""
        self.running = False

    def is_running(self) -> bool:
        """Check if tool preload is running."""
        return self.running


_tool_preload_service: ToolPreloadService | None = None


def get_tool_preload_service() -> ToolPreloadService | None:
    """Get the tool preload service instance if preload is enabled."""
    global _tool_preload_service
    config = get_default_config()

    if not config.preload_tools:
        _logger.info("Tool preload is disabled in configuration")
        return None

    if _tool_preload_service is None:
        _tool_preload_service = ToolPreloadService()
    return _tool_preload_service
