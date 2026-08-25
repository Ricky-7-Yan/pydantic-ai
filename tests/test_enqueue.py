from __future__ import annotations

import threading
from collections.abc import Callable
from types import NoneType
from typing import Any

import pytest

from pydantic_ai._enqueue import PendingMessage, PendingMessagePriority, PendingMessageQueue
from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


class _BlockingPendingMessageQueue(PendingMessageQueue):
    def __init__(
        self, messages: list[PendingMessage], drain_started: threading.Event, release_drain: threading.Event
    ) -> None:
        super().__init__(messages)
        self._drain_started = drain_started
        self._release_drain = release_drain

    def _pop_priority(self, priority: PendingMessagePriority) -> list[PendingMessage]:
        self._drain_started.set()
        assert self._release_drain.wait(timeout=1)
        return super()._pop_priority(priority)


def _race_enqueue_with_drain(
    ctx: RunContext[Any],
    drain: Callable[[], object],
    drain_started: threading.Event,
    release_drain: threading.Event,
) -> list[UserError]:
    enqueue_started = threading.Event()
    enqueue_errors: list[UserError] = []

    def enqueue() -> None:
        enqueue_started.set()
        try:
            ctx.enqueue('from sync background work')
        except UserError as error:
            enqueue_errors.append(error)

    drain_thread = threading.Thread(target=drain)
    drain_thread.start()
    assert drain_started.wait(timeout=1)

    enqueue_thread = threading.Thread(target=enqueue)
    enqueue_thread.start()
    assert enqueue_started.wait(timeout=1)
    assert enqueue_thread.is_alive()

    release_drain.set()
    drain_thread.join(timeout=1)
    enqueue_thread.join(timeout=1)
    assert not drain_thread.is_alive()
    assert not enqueue_thread.is_alive()
    return enqueue_errors


async def test_enqueue_after_run_ends_raises():
    """A retained tool context cannot enqueue into a run that has ended."""
    captured_ctx: RunContext[object] | None = None

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(message, ModelResponse) for message in messages):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='capture_context', args='{}')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def capture_context(ctx: RunContext[object]) -> str:
        nonlocal captured_ctx
        captured_ctx = ctx
        return 'captured'

    await agent.run('capture it')

    assert captured_ctx is not None
    with pytest.raises(UserError, match='run has ended'):
        captured_ctx.enqueue('too late')


async def test_enqueue_after_run_setup_fails_raises():
    """A context retained by `for_run` is closed if later setup fails."""
    captured_ctx: RunContext[None] | None = None

    class FailingCapability(AbstractCapability[None]):
        def get_model(self) -> TestModel:
            return TestModel()

        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            nonlocal captured_ctx
            captured_ctx = ctx
            return self

        def get_wrapper_toolset(self, toolset: AbstractToolset[None]) -> AbstractToolset[None]:
            raise RuntimeError('setup failed')

    agent = Agent(deps_type=NoneType, capabilities=[FailingCapability()])

    with pytest.raises(RuntimeError, match='setup failed'):
        await agent.run('hello')

    assert captured_ctx is not None
    with pytest.raises(UserError, match='run has ended'):
        captured_ctx.enqueue('too late')


def test_sync_enqueue_waits_for_pending_message_drain():
    """A sync enqueue cannot be lost while the pending-message queue is drained."""
    drain_started = threading.Event()
    release_drain = threading.Event()
    pending_messages: list[PendingMessage] = []
    pending_message_queue = _BlockingPendingMessageQueue(pending_messages, drain_started, release_drain)
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        pending_messages=pending_messages,
        _pending_message_queue=pending_message_queue,
    )

    enqueue_errors = _race_enqueue_with_drain(
        ctx, lambda: pending_message_queue.pop_priority('asap'), drain_started, release_drain
    )

    assert not enqueue_errors
    assert len(pending_messages) == 1


def test_sync_enqueue_racing_run_end_is_rejected():
    """Final queue inspection and closure are atomic with a sync enqueue."""
    drain_started = threading.Event()
    release_drain = threading.Event()
    pending_messages: list[PendingMessage] = []
    pending_message_queue = _BlockingPendingMessageQueue(pending_messages, drain_started, release_drain)
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        pending_messages=pending_messages,
        _pending_message_queue=pending_message_queue,
    )
    enqueue_errors = _race_enqueue_with_drain(ctx, pending_message_queue.drain_at_end, drain_started, release_drain)

    assert len(enqueue_errors) == 1
    assert not pending_messages
