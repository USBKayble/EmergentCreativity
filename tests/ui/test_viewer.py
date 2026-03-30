"""
tests/ui/test_viewer.py
=======================
Unit tests for the Viewer module.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.emergent_creativity.ui.viewer import SimViewer as Viewer, Action, MANUAL_ACTION_MAP

@pytest.fixture
def mock_env():
    env = MagicMock()
    env.reset.return_value = ({"vitals": [0.5, 0.5, 0.5, 0.5]}, {})
    env.step.return_value = ({"vitals": [0.5, 0.5, 0.5, 0.5]}, 1.0, False, False, {})
    return env

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_viewer_init(mock_pygame, mock_env):
    viewer = Viewer(env=mock_env)
    assert viewer.env == mock_env
    assert viewer._paused is False
    assert viewer._manual_mode is True

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_handle_keydown_quit(mock_pygame, mock_env):
    viewer = Viewer(env=mock_env)
    viewer._handle_keydown(mock_pygame.K_q)
    assert viewer._running is False

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_handle_keydown_pause(mock_pygame, mock_env):
    viewer = Viewer(env=mock_env)
    viewer._handle_keydown(mock_pygame.K_SPACE)
    assert viewer._paused is True
    viewer._handle_keydown(mock_pygame.K_SPACE)
    assert viewer._paused is False

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_handle_keydown_reset(mock_pygame, mock_env):
    viewer = Viewer(env=mock_env)
    viewer._total_reward = 10.0
    viewer._episode_step = 100
    viewer._handle_keydown(mock_pygame.K_r)
    assert mock_env.reset.called
    assert viewer._total_reward == 0.0
    assert viewer._episode_step == 0

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_handle_keydown_manual_mode_toggle(mock_pygame, mock_env):
    viewer = Viewer(env=mock_env)
    initial_mode = viewer._manual_mode
    viewer._handle_keydown(mock_pygame.K_i)
    assert viewer._manual_mode == (not initial_mode)

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_handle_keydown_continuous_mode_toggle(mock_pygame, mock_env):
    viewer = Viewer(env=mock_env)
    initial_mode = viewer._continuous_mode
    viewer._handle_keydown(mock_pygame.K_c)
    assert viewer._continuous_mode == (not initial_mode)

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_handle_keydown_manual_actions(mock_pygame, mock_env):
    viewer = Viewer(env=mock_env)
    for key, action in MANUAL_ACTION_MAP.items():
        viewer._handle_keydown(key)
        assert viewer._manual_action == action

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_nn_act_with_legacy_agent(mock_pygame, mock_env):
    mock_agent = MagicMock()
    mock_agent.return_value = Action.MOVE_FORWARD
    viewer = Viewer(env=mock_env, nn_agent=mock_agent)
    action = viewer._nn_act(None, None)
    assert action == Action.MOVE_FORWARD

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_nn_act_with_legacy_agent_returns_tuple(mock_pygame, mock_env):
    mock_agent = MagicMock()
    mock_agent.return_value = (Action.TURN_LEFT, "some_state")
    viewer = Viewer(env=mock_env, nn_agent=mock_agent)
    action = viewer._nn_act(None, None)
    assert action == Action.TURN_LEFT

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_nn_act_with_no_agent(mock_pygame, mock_env):
    viewer = Viewer(env=mock_env)
    action = viewer._nn_act(None, None)
    assert action == Action.IDLE

@patch("src.emergent_creativity.ui.viewer.pygame")
def test_nn_act_with_exception(mock_pygame, mock_env):
    mock_agent = MagicMock(side_effect=Exception("Agent error"))
    viewer = Viewer(env=mock_env, nn_agent=mock_agent)
    action = viewer._nn_act(None, None)
    assert action == Action.IDLE
