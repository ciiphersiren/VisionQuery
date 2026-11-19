from dronekit import vlm


def test_parse_int_is_strict():
    assert vlm.parse_int("12") == 12 and vlm.parse_int(" 12.\n") == 12 and vlm.parse_int("0") == 0
    for bad in ("about 12", "12 cars", "There are 12.", "", None, "1,2", "-3"):
        assert vlm.parse_int(bad) is None


def test_parse_points_distinct_and_valid():
    assert vlm.parse_points("10,20\n30,40\n10,20\n") == 2
    assert vlm.parse_points("(10, 20)\n30,40\n30,") == 2           # last line cut by the token limit
    assert vlm.parse_points("NONE") == 0 and vlm.parse_points("none.") == 0
    assert vlm.parse_points("1500,20") is None and vlm.parse_points("I see cars") is None and vlm.parse_points("") is None


def test_prompts_name_the_target_and_boxes():
    assert "cars" in vlm.prompt_direct("car") and "integer" in vlm.prompt_direct("car")
    p = vlm.prompt_grounded("bus", [(1, 2, 3, 4, 0.5)])
    assert "1: 1,2,3,4 score 0.50" in p and "1 candidate" in p
    assert "(none)" in vlm.prompt_grounded("bus", [])



def test_parse_points_lenient_accepts_boxes():
    assert vlm.parse_points_lenient("```plaintext\n365,237,491,348 car\n712,112,758,145 car\n365,237,491,348 car\n```") == 2
    assert vlm.parse_points_lenient("10,20\n30,40") == 2 and vlm.parse_points_lenient("NONE") == 0
    assert vlm.parse_points_lenient("1,2,3,4 car\n5,6,7,") == 1                  # last line cut by the token limit
    assert vlm.parse_points_lenient("246,58,355,107 x,y: 390,100") is None and vlm.parse_points_lenient("I see cars") is None
    assert vlm.parse_points("365,237,491,348 car") is None                        # the pre-registered parser stays strict
