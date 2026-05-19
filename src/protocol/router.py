"""Lightweight in-process message bus used by the LangGraph orchestrator.

The router intentionally has no transport layer — MARGO phases run within a
single process. Its only jobs are (a) fan-out, (b) recording an immutable
trace for the demo UI and CADR metric, and (c) optional subscribers for the
WebSocket trace endpoint of the web demo.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Iterable

from protocol.messages import Message

log = logging.getLogger(__name__)


Subscriber = Callable[[Message], None]


class MessageBus:
    """A pub/sub bus with persistent trace.

    Agents do not normally call the bus directly — the lifecycle nodes do.
    """

    def __init__(self) -> None:
        self._subs: dict[str, list[Subscriber]] = defaultdict(list)
        self._global: list[Subscriber] = []
        self.trace: list[Message] = []

    # ------------------------------------------------------------------ #
    # Wiring                                                              #
    # ------------------------------------------------------------------ #

    def subscribe(self, agent_id: str, fn: Subscriber) -> None:
        self._subs[agent_id].append(fn)

    def subscribe_all(self, fn: Subscriber) -> None:
        """Useful for the web demo's WebSocket forwarder."""
        self._global.append(fn)

    # ------------------------------------------------------------------ #
    # Delivery                                                            #
    # ------------------------------------------------------------------ #

    def publish(self, message: Message) -> None:
        self.trace.append(message)
        log.debug(
            "msg %-12s %s → %s", message.type.value, message.sender, message.receivers
        )
        for fn in self._global:
            try:
                fn(message)
            except Exception:  # pragma: no cover — observers must not break the run
                log.exception("global subscriber raised; continuing")

        for receiver in message.receivers:
            for fn in self._subs.get(receiver, ()):
                try:
                    fn(message)
                except Exception:
                    log.exception("subscriber %s raised; continuing", receiver)

    def history(self) -> Iterable[Message]:
        return tuple(self.trace)

    def clear(self) -> None:
        self.trace.clear()
