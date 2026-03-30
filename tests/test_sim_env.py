import pytest
from src.emergent_creativity.sim_env import TenantEnv as ApartmentEnv

def test_apartment_env_step_without_reset_raises_error():
    env = ApartmentEnv(gui=False)
    with pytest.raises(RuntimeError, match=r"Environment not built\. Call reset\(\) first\."):
        env.step(0)
