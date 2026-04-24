import pytest
from unittest.mock import MagicMock, PropertyMock
import numpy as np

from src.emergent_creativity.sim_env import TenantEnv
from src.emergent_creativity.sim_env import TenantEnv as ApartmentEnv


class TestTenantEnvStep:
    def test_step_uninitialized(self):
        env = TenantEnv()
        with pytest.raises(
            RuntimeError, match="Environment not built. Call reset\\(\\) first."
        ):
            env.step(0)

    def test_apartment_env_step_without_reset_raises_error(self):
        env = ApartmentEnv(gui=False)
        with pytest.raises(
            RuntimeError, match=r"Environment not built\. Call reset\(\) first\."
        ):
            env.step(0)
            env.step(0)

    def test_step_success(self, monkeypatch):
        # Create a mock environment without starting pybullet or loading real config
        env = TenantEnv()

        # Mock the internal components
        env._tenant = MagicMock()
        env._world = MagicMock()  # This provides `env.physics`
        env._apartment = MagicMock()
        env._registry = MagicMock()
        env._evaluator = MagicMock()

        # Set up mock returns
        env._evaluator.evaluate.return_value = (10.0, {"eat_food": 10.0})
        env._evaluator.is_terminal.return_value = False

        env._tenant.vitals.to_array.return_value = np.array([0.5, 0.5, 0.5, 0.5])
        env._tenant.events = ["ate_food"]
        env._tenant.total_steps = 42

        env._registry.mess_count.return_value = 2

        # Mock _get_obs to avoid sensory/observation computation
        dummy_obs = {"vision": np.zeros((64, 64, 3))}
        monkeypatch.setattr(env, "_get_obs", MagicMock(return_value=dummy_obs))

        # Perform the step
        action = 1
        obs, reward, terminated, truncated, info = env.step(action)

        # Assert correct methods were called
        env._tenant.step.assert_called_once_with(action)
        env._world.step.assert_called_once_with()
        env._apartment.sync_registry.assert_called_once_with()
        env._evaluator.evaluate.assert_called_once_with(env._tenant, env._registry)
        env._evaluator.is_terminal.assert_called_once_with(env._tenant)
        env._get_obs.assert_called_once_with()

        # Assert correct returns
        assert obs == dummy_obs
        assert reward == 10.0
        assert terminated is False
        assert truncated is False
        assert info == {
            "reward_breakdown": {"eat_food": 10.0},
            "vitals": [0.5, 0.5, 0.5, 0.5],
            "events": ["ate_food"],
            "mess_count": 2,
            "step": 42,
        }

    def test_build_yaml_parsing_error_fallback(self, monkeypatch):
        # Create environment
        env = TenantEnv()

        # Mock dependencies to avoid actual instantiation during test
        from src.emergent_creativity.sim_env import CameraSpec
        mock_camera_spec = MagicMock(spec=CameraSpec)
        monkeypatch.setattr("src.emergent_creativity.sim_env.CameraSpec", mock_camera_spec)

        mock_physics_world = MagicMock()
        monkeypatch.setattr("src.emergent_creativity.sim_env.PhysicsWorld", mock_physics_world)

        mock_apartment = MagicMock()
        monkeypatch.setattr("src.emergent_creativity.sim_env.Apartment", mock_apartment)

        mock_sensory_suite = MagicMock()
        monkeypatch.setattr("src.emergent_creativity.sim_env.SensorySuite", mock_sensory_suite)

        mock_tenant = MagicMock()
        monkeypatch.setattr("src.emergent_creativity.sim_env.Tenant", mock_tenant)

        # Mock 'open' to raise an Exception, simulating a YAML parsing error or missing file
        def mock_open(*args, **kwargs):
            raise Exception("Simulated YAML exception")
        monkeypatch.setattr("builtins.open", mock_open)

        # Import constants to assert against
        from src.emergent_creativity.environment.senses import VISION_W, VISION_H

        # Call the private build method
        env._build(seed=0)

        # Assert that CameraSpec received the default values since obs_cfg was fallback {}
        mock_camera_spec.assert_called_once_with(width=VISION_W, height=VISION_H, fov=90.0)

        # Assert that SensorySuite received the empty obs_cfg dictionary
        # It's called with: SensorySuite(self._world, self._registry, agent_body_id=-1, cfg=obs_cfg)
        mock_sensory_suite.assert_called_once()
        assert mock_sensory_suite.call_args.kwargs.get('cfg') == {}

        # Assert that Tenant received the empty vitals_cfg dictionary
        # It's called with: Tenant(self._world, self._registry, self._sensors, vitals_cfg=vitals_cfg)
        mock_tenant.assert_called_once()
        assert mock_tenant.call_args.kwargs.get('vitals_cfg') == {}
