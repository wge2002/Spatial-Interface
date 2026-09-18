"""Behavioral regression for the terminal-state feedback defect (seed3 of
astra_autofallback_20260911_0325).

record_sim writes verdict.json on a normal terminal episode and only then closes
the UI, so the teardown races the tool call still in flight. Observed: execute's
Page.wait_for_function raised "closed", end_episode's Page.screenshot raised
"closed", and the official verdict was success=true -- the model was told nothing
about a finished, won episode.

These tests drive the real ExecuteTool / EndEpisodeTool / read_current_episode_verdict
against controllable page stand-ins, and assert on what the agent receives. No
assertions on source text.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import pytest
from playwright.async_api import Error as PlaywrightError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spatial_interface.mcp_server as m  # noqa: E402

# The recovery helpers are new. Resolve them dynamically so that on a build
# lacking the fix each test fails on the behavior it guards instead of the whole
# module erroring at import.
_read_verdict = getattr(m, "read_current_episode_verdict", lambda ctx: None)


def test_module_resolves_to_this_worktree():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(m.__file__).startswith(root)


# --- Test doubles -------------------------------------------------------------


class _ClosedError(PlaywrightError):
    """Stands in for Playwright's error on a torn-down page.

    Deliberately NOT PlaywrightTimeoutError and not a bespoke class the code
    special-cases: recognition must come from the closure itself (page.is_closed
    or the message), which is all the real error offers.
    """


CLOSED_MSG = "Target page, context or browser has been closed"


class _FakeLocator:
    def __init__(self, page, selector):
        self.page, self.selector = page, selector

    async def click(self):
        self.page._maybe_fail("click")
        self.page.clicks.append(self.selector)


class _FakePage:
    """Page stand-in that can close at a chosen stage of the tool call.

    close_at: None | "wait" | "pose" | "screenshot" | "already" -- "already" means
    the page was closed before the tool was even entered.
    """

    def __init__(self, *, completes=True, close_at=None, raise_plain_at=None):
        self.completes = completes
        self.close_at = close_at
        self.raise_plain_at = raise_plain_at
        self.closed = close_at == "already"
        self.clicks: list[str] = []
        self.waits = 0

    def is_closed(self):
        return self.closed

    def locator(self, selector):
        return _FakeLocator(self, selector)

    def _maybe_fail(self, stage):
        if self.raise_plain_at == stage:
            raise RuntimeError("Element is not visible")  # ordinary Playwright fault
        if self.close_at == stage:
            self.closed = True
            raise _ClosedError(CLOSED_MSG)
        if self.closed:
            raise _ClosedError(CLOSED_MSG)

    async def evaluate(self, script, arg=None):
        self._maybe_fail("evaluate")
        return 0

    async def wait_for_function(self, script, arg=None, timeout=None):
        self.waits += 1
        self._maybe_fail("wait")
        if not self.completes:
            raise m.PlaywrightTimeoutError("no completion signal")
        return True

    async def screenshot(self, **kw):
        self._maybe_fail("screenshot")
        return b"jpeg"


class _Ctx:
    """ToolContext stand-in wired to a _FakePage, with the real page-touching
    helpers routed through it so a closure at any stage is reproduced."""

    def __init__(self, page, screenshots_dir=None):
        self.page = page
        self.screenshots_dir = screenshots_dir
        self.name = "execute_waypoint"

    async def gripper_pose_text(self):
        self.page._maybe_fail("pose")
        return "Current Target Gripper Pose: <pose>"

    async def snap(self, text=None):
        img = await self.page.screenshot()
        return [m.types.TextContent(type="text", text=text or ""),
                m.types.ImageContent(type="image", data="x", mimeType="image/jpeg")]


def _texts(content):
    return "\n".join(c.text for c in content if getattr(c, "type", None) == "text")


def _has_image(content):
    return any(getattr(c, "type", None) == "image" for c in content)


# --- Episode fixtures ---------------------------------------------------------


def _make_episode(tmp_path, *, success=True, name="demo00000", write=True,
                  body=None, mtime=None, screenshots_dir=None, timed_out=False):
    """Build a demo folder laid out like record_sim's: <demo>/screenshots/ plus
    the <demo>/verdict.json sidecar."""
    demo = tmp_path / name
    screens = demo / "screenshots"
    screens.mkdir(parents=True)
    sidecar = demo / "verdict.json"
    if write:
        record = body if body is not None else {
            "task": "libero_goal/8",
            "language": "put the bowl on the plate",
            "success": success,
            "timed_out": timed_out,
            "demo_folder": str(demo),
            "screenshots_dir": screenshots_dir or str(screens),
        }
        sidecar.write_text(record if isinstance(record, str) else json.dumps(record))
        if mtime is not None:
            os.utime(sidecar, (mtime, mtime))
    return str(screens)


@pytest.fixture(autouse=True)
def _fast_and_isolated(monkeypatch):
    """No real repaint delay, and no live sim: success.json is unreachable in the
    unit environment, which is also the honest state after a teardown."""
    monkeypatch.setattr(m, "FRAME_DELAY_S", 0.0)

    async def _no_sim(path):
        return None

    monkeypatch.setattr(m, "_fetch_sim_json", _no_sim)
    # This episode's sidecar is written after the server starts; the fixtures
    # below write it "now", so anchor the start marker in the past. Guarded with
    # raising=False so that on a build without the fix these tests fail on
    # BEHAVIOR (the missing terminal reply) rather than erroring in setup.
    monkeypatch.setattr(m, "_SERVER_START_TIME", time.time() - 60, raising=False)


# --- read_current_episode_verdict: only trustworthy current records ----------


def test_verdict_is_read_for_the_current_episode(tmp_path):
    ctx = _Ctx(_FakePage(), _make_episode(tmp_path, success=True))
    record = _read_verdict(ctx)
    assert record is not None and record["success"] is True


def test_verdict_is_read_through_a_symlinked_screenshots_dir(tmp_path):
    """Path identity must be compared on realpaths, or a symlinked episode dir
    fails the ownership check and a real terminal state is lost."""
    real = _make_episode(tmp_path, success=True)
    link = tmp_path / "linked_screens"
    link.symlink_to(real)
    assert _read_verdict(_Ctx(_FakePage(), str(link))) is not None


def test_missing_sidecar_yields_no_verdict(tmp_path):
    ctx = _Ctx(_FakePage(), _make_episode(tmp_path, write=False))
    assert _read_verdict(ctx) is None


def test_malformed_sidecar_yields_no_verdict(tmp_path):
    ctx = _Ctx(_FakePage(), _make_episode(tmp_path, body="{not json"))
    assert _read_verdict(ctx) is None


def test_non_object_sidecar_yields_no_verdict(tmp_path):
    ctx = _Ctx(_FakePage(), _make_episode(tmp_path, body=json.dumps([1, 2, 3])))
    assert _read_verdict(ctx) is None


def test_stale_sidecar_from_a_previous_episode_is_refused(tmp_path):
    """A sidecar older than this server process belongs to an earlier episode."""
    screens = _make_episode(tmp_path, success=True,
                            mtime=time.time() - 3600)  # before _SERVER_START_TIME
    assert _read_verdict(_Ctx(_FakePage(), screens)) is None


def test_other_episodes_sidecar_is_refused(tmp_path):
    """Cross-episode guard: our screenshots dir with another episode's record.

    Without the identity check, a stray or hand-copied verdict.json could report
    a foreign success as this episode's outcome.
    """
    other = tmp_path / "demo99999" / "screenshots"
    other.mkdir(parents=True)
    screens = _make_episode(tmp_path, success=True, screenshots_dir=str(other))
    assert _read_verdict(_Ctx(_FakePage(), screens)) is None


def test_sidecar_claiming_a_different_demo_folder_is_refused(tmp_path):
    screens = _make_episode(tmp_path, body=json.dumps({
        "task": "libero_goal/8", "success": True,
        "demo_folder": str(tmp_path / "somewhere_else"),
        "screenshots_dir": str(tmp_path / "demo00000" / "screenshots"),
    }))
    assert _read_verdict(_Ctx(_FakePage(), screens)) is None


@pytest.mark.parametrize("grade", [None, "true", 1, "success"])
def test_ungraded_or_non_boolean_grade_yields_no_verdict(tmp_path, grade):
    """success must be an explicit boolean. A truthy string is not a grade."""
    demo = tmp_path / "demo00000"
    screens = demo / "screenshots"
    screens.mkdir(parents=True)
    (demo / "verdict.json").write_text(json.dumps({
        "task": "libero_goal/8", "success": grade,
        "demo_folder": str(demo), "screenshots_dir": str(screens),
    }))
    assert _read_verdict(_Ctx(_FakePage(), str(screens))) is None


def test_no_screenshots_dir_means_no_verdict_and_no_filesystem_search(tmp_path):
    """With no episode identity from /env.json there is nothing to trust, and the
    code must not go looking for an arbitrary verdict elsewhere."""
    _make_episode(tmp_path, success=True)
    assert _read_verdict(_Ctx(_FakePage(), None)) is None


# --- execute_waypoint: success auto-close ------------------------------------


@pytest.mark.parametrize("close_at", ["wait", "pose", "screenshot"])
def test_success_autoclose_reports_success_terminal_state(tmp_path, close_at):
    """The actual defect. Whichever stage the teardown lands on, the reply must
    state the episode succeeded and is over."""
    screens = _make_episode(tmp_path, success=True)
    page = _FakePage(close_at=close_at)
    ctx = _Ctx(page, screens)
    text = _texts(asyncio.run(m.ExecuteTool()(ctx, {})))

    assert "TASK SUCCEEDED" in text
    assert "SUCCESS" in text
    assert "libero_goal/8" in text
    assert "stop here" in text.lower()
    assert "UNKNOWN" not in text


@pytest.mark.parametrize("close_at", ["wait", "pose", "screenshot"])
def test_terminal_reply_actuates_exactly_once(tmp_path, close_at):
    """The #btn-record click IS the actuation; a closed page must not be clicked
    again, and the reply must warn off a retry."""
    screens = _make_episode(tmp_path, success=True)
    page = _FakePage(close_at=close_at)
    text = _texts(asyncio.run(m.ExecuteTool()(_Ctx(page, screens), {})))
    assert page.clicks == ["#btn-record"]
    assert "do NOT retry" in text or "do not retry" in text.lower()


@pytest.mark.parametrize("close_at", ["wait", "pose", "screenshot"])
def test_terminal_reply_returns_text_without_a_fabricated_screenshot(tmp_path, close_at):
    """The final frame is genuinely gone. Text is allowed; a fake observation is
    not."""
    screens = _make_episode(tmp_path, success=True)
    content = asyncio.run(m.ExecuteTool()(_Ctx(_FakePage(close_at=close_at), screens), {}))
    assert not _has_image(content)
    assert _texts(content).strip()


def test_failure_terminal_state_is_not_reported_as_success(tmp_path):
    """Not every close is a win. A graded failure must read as FAILURE."""
    screens = _make_episode(tmp_path, success=False)
    text = _texts(asyncio.run(m.ExecuteTool()(_Ctx(_FakePage(close_at="wait"), screens), {})))
    assert "FAILURE" in text
    assert "TASK SUCCEEDED" not in text
    assert "SUCCESS" not in text.replace("TASK SUCCEEDED", "")


def test_timed_out_terminal_state_is_reported_as_failure(tmp_path):
    screens = _make_episode(tmp_path, success=False, timed_out=True)
    text = _texts(asyncio.run(m.ExecuteTool()(_Ctx(_FakePage(close_at="wait"), screens), {})))
    assert "FAILURE" in text and "horizon" in text
    assert "TASK SUCCEEDED" not in text


@pytest.mark.parametrize("close_at", ["wait", "pose", "screenshot"])
def test_close_without_a_trustworthy_record_is_unknown_not_success(tmp_path, close_at):
    """A closed connection is not evidence of success: with no usable record the
    reply must be an error with an UNKNOWN outcome."""
    screens = _make_episode(tmp_path, write=False)
    text = _texts(asyncio.run(m.ExecuteTool()(_Ctx(_FakePage(close_at=close_at), screens), {})))
    assert "UNKNOWN" in text
    assert "ERROR" in text
    assert "TASK SUCCEEDED" not in text


def test_stale_record_close_is_unknown_not_the_old_verdict(tmp_path):
    """The strongest wrong-answer case: an old success sidecar must not be
    reported for a fresh closure."""
    screens = _make_episode(tmp_path, success=True, mtime=time.time() - 3600)
    text = _texts(asyncio.run(m.ExecuteTool()(_Ctx(_FakePage(close_at="wait"), screens), {})))
    assert "UNKNOWN" in text and "TASK SUCCEEDED" not in text


def test_terminal_reply_does_not_leak_absolute_paths(tmp_path):
    """Hidden eval data must stay hidden: no episode paths in the model's text."""
    screens = _make_episode(tmp_path, success=True)
    text = _texts(asyncio.run(m.ExecuteTool()(_Ctx(_FakePage(close_at="wait"), screens), {})))
    assert str(tmp_path) not in text
    assert "verdict.json" not in text
    assert "screenshots" not in text


# --- ordinary errors must not be reshaped into terminal states ----------------


@pytest.mark.parametrize("stage", ["wait", "pose", "screenshot"])
def test_plain_playwright_error_still_propagates(tmp_path, stage):
    """A non-closure fault is a bug to surface, not a finished episode. It must
    not be swallowed even when a success sidecar happens to exist."""
    screens = _make_episode(tmp_path, success=True)
    page = _FakePage(raise_plain_at=stage)
    with pytest.raises(RuntimeError, match="not visible"):
        asyncio.run(m.ExecuteTool()(_Ctx(page, screens), {}))


def test_timeout_path_is_unchanged_by_the_terminal_handling(tmp_path):
    """The existing timeout contract must survive: submitted/unknown, one click,
    and a real screenshot to observe with."""
    screens = _make_episode(tmp_path, write=False)
    page = _FakePage(completes=False)
    content = asyncio.run(m.ExecuteTool()(_Ctx(page, screens), {}))
    text = _texts(content)
    assert "SUBMITTED" in text and "UNKNOWN" in text
    assert page.clicks == ["#btn-record"]
    assert _has_image(content)


def test_normal_success_path_is_unchanged(tmp_path):
    """The ordinary completed waypoint must read exactly as before."""
    screens = _make_episode(tmp_path, success=True)
    page = _FakePage(completes=True)
    content = asyncio.run(m.ExecuteTool()(_Ctx(page, screens), {}))
    assert _texts(content) == (
        "Executed current target and recorded waypoint.\n"
        "Current Target Gripper Pose: <pose>"
    )
    assert _has_image(content)
    assert page.clicks == ["#btn-record"]


# --- end_episode: no side effects on a closed episode ------------------------


def test_end_episode_on_closed_page_reports_terminal_state_without_clicking(tmp_path):
    screens = _make_episode(tmp_path, success=True)
    page = _FakePage(close_at="already")
    content = asyncio.run(m.EndEpisodeTool()(_Ctx(page, screens), {}))
    text = _texts(content)
    assert page.clicks == [], "must not click #btn-end on a closed UI"
    assert "TASK SUCCEEDED" in text and "SUCCESS" in text
    assert not _has_image(content)


def test_end_episode_on_closed_page_without_record_is_unknown(tmp_path):
    screens = _make_episode(tmp_path, write=False)
    page = _FakePage(close_at="already")
    text = _texts(asyncio.run(m.EndEpisodeTool()(_Ctx(page, screens), {})))
    assert "UNKNOWN" in text and "TASK SUCCEEDED" not in text
    assert page.clicks == []


def test_end_episode_closing_mid_screenshot_reports_terminal_state(tmp_path):
    screens = _make_episode(tmp_path, success=True)
    page = _FakePage(close_at="screenshot")
    content = asyncio.run(m.EndEpisodeTool()(_Ctx(page, screens), {}))
    assert "TASK SUCCEEDED" in _texts(content)
    assert page.clicks == []


def test_repeated_end_episode_does_not_click_twice(tmp_path):
    """First call ends the episode (UI closes); the second must be a pure report."""
    screens = _make_episode(tmp_path, success=True)
    page = _FakePage()
    ctx = _Ctx(page, screens)
    asyncio.run(m.EndEpisodeTool()(ctx, {}))
    assert page.clicks == ["#btn-end"]
    page.closed = True  # the UI tore down, as it does after ending
    asyncio.run(m.EndEpisodeTool()(ctx, {}))
    assert page.clicks == ["#btn-end"], "second end_episode must not re-click"


def test_end_episode_plain_click_error_propagates(tmp_path):
    """A non-closure click failure is a real fault; only closures are tolerated."""
    screens = _make_episode(tmp_path, success=True)

    class _BadClick(_FakePage):
        def locator(self, selector):
            page = self

            class _L:
                async def click(self):
                    raise RuntimeError("Element is not visible")

            return _L()

    with pytest.raises(RuntimeError, match="not visible"):
        asyncio.run(m.EndEpisodeTool()(_Ctx(_BadClick(), screens), {}))


def test_end_episode_normal_path_is_unchanged(tmp_path, monkeypatch):
    """With a live sim the ordinary reply and single click must be untouched."""
    async def _live(path):
        assert path == "success.json"
        return {"success": True, "task": "libero_goal/8"}

    monkeypatch.setattr(m, "_fetch_sim_json", _live)
    screens = _make_episode(tmp_path, write=False)
    page = _FakePage()
    content = asyncio.run(m.EndEpisodeTool()(_Ctx(page, screens), {}))
    assert _texts(content) == (
        "Episode ended and saved. Task verdict: SUCCESS (libero_goal/8)."
    )
    assert page.clicks == ["#btn-end"]
    assert _has_image(content)


@pytest.mark.parametrize("close_at", ["already", "evaluate", "click"])
def test_terminal_before_execution_reports_verdict_without_actuation(tmp_path, close_at):
    screens = _make_episode(tmp_path, success=True)
    page = _FakePage(close_at=close_at)
    result = asyncio.run(m.ExecuteTool()(_Ctx(page, screens), {}))
    assert "TASK SUCCEEDED" in _texts(result)
    assert page.clicks == []
    assert "already been submitted" not in _texts(result)


@pytest.mark.parametrize("stage", ["evaluate", "click"])
def test_execution_preparation_error_propagates(tmp_path, stage):
    screens = _make_episode(tmp_path, success=True)
    page = _FakePage(raise_plain_at=stage)
    with pytest.raises(RuntimeError, match="not visible"):
        asyncio.run(m.ExecuteTool()(_Ctx(page, screens), {}))


@pytest.mark.parametrize("has_verdict", [True, False])
def test_end_click_closed_uses_durable_verdict_not_stale_live_flag(tmp_path, monkeypatch, has_verdict):
    async def _stale_live(path):
        return {"success": False, "task": "libero_goal/8"}
    monkeypatch.setattr(m, "_fetch_sim_json", _stale_live)
    screens = _make_episode(tmp_path, success=True, write=has_verdict)
    page = _FakePage(close_at="click")
    result = asyncio.run(m.EndEpisodeTool()(_Ctx(page, screens), {}))
    text = _texts(result)
    assert "FAILURE" not in text
    assert ("TASK SUCCEEDED" in text) is has_verdict
    assert ("UNKNOWN" in text) is (not has_verdict)
    assert page.clicks == []
