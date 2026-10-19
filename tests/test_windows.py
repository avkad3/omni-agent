from omni_agent.windows import find_window

WINDOWS = {
    1: "● .env - competitor-tool - Visual Studio Code",
    2: "Untitled - Notepad",
    3: "Calculator",
    4: "Home - File Explorer",
    5: "Settings",
    6: "Untitled - Paint",
}


def test_find_window_matches_app_names_not_substrings_of_words():
    assert find_window("Notepad", WINDOWS) == 2
    assert find_window("calculator", WINDOWS) == 3
    assert find_window("File Explorer", WINDOWS) == 4
    assert find_window("Setting", WINDOWS) == 5
    assert find_window("Paint", WINDOWS) == 6
    assert find_window("Code", WINDOWS) == 1
    assert find_window("Clock", WINDOWS) is None
    assert find_window("tool", {9: "competitor-tool"}) is None  # 'tool' inside a word is not a match
