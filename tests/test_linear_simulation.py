from tests.simulation.linear_harness import LinearSimulationHarness, available_scenarios
import pytest

@pytest.mark.parametrize("scenario", available_scenarios(), ids=lambda item:item.scenario_id)
def test_linear_scenario(scenario):
    assert LinearSimulationHarness().run(scenario.scenario_id)["outcome"] == "passed"
