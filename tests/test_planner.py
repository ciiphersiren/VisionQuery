import pytest

from skyquery.planner.policy import Planner
from skyquery.planner.rules import plan_rules

H = ["pedestrian", "people"]

CASES = [
    ("how many cars are in the left half?", "count", [["car"]], "left half", []),
    ("Are there any buses?", "exists", [["bus"]], "whole", []),
    ("is anyone sitting in the top right corner?", "exists", [["people"]], "top-right quadrant", []),
    ("count the tricycles with awnings", "count", [["awning-tricycle"]], "whole", []),   # regression test
    ("how many fire trucks are there?", "count", [], "whole", ["fire trucks"]),          # must not silently map
    ("how many people?", "count", [H], "whole", []),
    ("are there more cars than vans in the bottom half?", "compare", [["car"], ["van"]], "bottom half", []),
    ("show me all the motorbikes", "locate", [["motor"]], "whole", []),
    ("detect any trucks", "locate", [["truck"]], "whole", []),                            # regression test
    ("how many dogs are in the center?", "count", [], "center", ["dog"]),
    ("number of pedestrians walking in the middle third", "count", [["pedestrian"]], "middle third", []),
    ("how many people riding bikes", "count", [["people"]], "whole", []),
    ("cars and buses in the lower right", "count", [["car"], ["bus"]], "bottom-right quadrant", []),
    ("is there an ambulance", "exists", [], "whole", ["ambulance"]),
    ("how many red cars on the right", "count", [], "right half", ["red cars"]),
    ("what's in this image?", "summary", [], "whole", []),
    ("how many vehicles in the upper-left quadrant", "count", [["car", "van", "truck", "bus", "motor", "tricycle", "awning-tricycle"]], "top-left quadrant", []),
]


@pytest.mark.parametrize("q,intent,targets,region,oov", CASES)
def test_rules(q, intent, targets, region, oov):
    p = plan_rules(q)
    assert p.intent == intent
    assert [t.classes for t in p.targets] == targets
    assert p.region == region
    assert [o.term for o in p.oov] == oov


def test_fire_truck_keeps_parent():
    p = plan_rules("how many fire trucks?")
    assert p.oov[0].nearest == ["truck"]


def test_unrecognised():
    assert plan_rules("hello there").source == "none"
    assert plan_rules("how many in the left half?").source == "none"


def test_rules_first_without_llm_falls_back_to_rules():
    pl = Planner("rules_first")
    pl.llm.api_key = None
    assert pl("how many unicorns?").oov[0].term == "unicorns"
    assert pl("how many cars?").targets[0].classes == ["car"]
