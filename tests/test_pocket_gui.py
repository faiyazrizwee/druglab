"""
Tests for DrugLab Interactive Graphical Pocket & Docking Box Tuner.
"""

import json
import urllib.request
import threading
import pytest
from unittest.mock import patch

from core.pocket_gui import (
    find_free_port,
    _PocketGUIHandler,
    _build_pocket_gui_html,
    launch_pocket_gui,
)
from core.pocket_detection import (
    interactive_pocket_selection,
    prompt_pocket_confirmation_with_selection,
    display_selected_pocket_summary,
)


@pytest.fixture
def sample_pockets():
    return [
        {
            "pocket_id": 0,
            "center": [110.983, 113.918, 115.921],
            "size": [30.0, 30.0, 30.0],
            "volume": 27000.0,
            "score": 850.5,
            "residues": ["ALA1", "GLY2", "VAL3"],
        },
        {
            "pocket_id": 1,
            "center": [105.0, 110.0, 112.0],
            "size": [25.0, 25.0, 25.0],
            "volume": 15625.0,
            "score": 620.0,
            "residues": ["LEU4", "ILE5"],
        },
    ]


def test_find_free_port():
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_build_pocket_gui_html(sample_pockets):
    html = _build_pocket_gui_html(
        protein_pdb_text="ATOM      1  N   ALA A   1      11.0  12.0  13.0",
        pockets=sample_pockets,
        initial_center=[110.983, 113.918, 115.921],
        initial_size=[30.0, 30.0, 30.0],
        max_box_dimension=30.0,
    )

    assert "DrugLab Pocket &amp; Docking Box Tuner" in html
    assert "Center:" in html
    assert "X = 110.983" in html
    assert "Y = 113.918" in html
    assert "Z = 115.921" in html
    assert "Size:" in html
    assert "X = 30.0" in html
    assert "3Dmol" in html
    assert "addBox" in html
    assert "addSphere" in html
    assert "e.key === \"Enter\"" in html
    assert "border-radius: 0" in html


def test_pocket_gui_http_endpoints(sample_pockets):
    import http.server

    port = find_free_port()
    server = http.server.HTTPServer(("127.0.0.1", port), _PocketGUIHandler)
    server.html_content = "<html><body>DrugLab Tuner Test</body></html>"
    server.selected_coordinates = ([110.983, 113.918, 115.921], [30.0, 30.0, 30.0])
    server.done_event = threading.Event()

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # GET /
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            assert "DrugLab Tuner Test" in body

        # GET /health
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data.get("status") == "ok"

        # POST /update_preview
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/update_preview",
            data=json.dumps({"center": [115.0, 118.0, 120.0], "size": [28.0, 28.0, 28.0]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
        assert server.selected_coordinates == ([115.0, 118.0, 120.0], [28.0, 28.0, 28.0])
        assert not server.done_event.is_set()

        # POST /select_pocket
        req_select = urllib.request.Request(
            f"http://127.0.0.1:{port}/select_pocket",
            data=json.dumps({"center": [112.5, 114.5, 116.5], "size": [26.0, 26.0, 26.0]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_select) as resp:
            assert resp.status == 200
            resp.read()
        assert server.selected_coordinates == ([112.5, 114.5, 116.5], [26.0, 26.0, 26.0])
        assert server.done_event.is_set()
    finally:
        server.shutdown()
        server.server_close()


def test_launch_pocket_gui_timeout(sample_pockets):
    with patch("webbrowser.open"):
        selected, method = launch_pocket_gui(
            sample_pockets,
            max_box_dimension=30.0,
            timeout=0.1,
        )

    assert method == "graphical_gui"
    assert "center" in selected
    assert "size" in selected
    assert selected["center"] == [110.983, 113.918, 115.921]
    assert selected["size"] == [30.0, 30.0, 30.0]


def test_interactive_pocket_selection_gui_option(sample_pockets):
    # Option 3 (GUI option: 2 pockets + 1 = 3)
    with patch("builtins.input", return_value="3"):
        with patch("core.pocket_gui.launch_pocket_gui") as mock_launch:
            mock_launch.return_value = (sample_pockets[0], "graphical_gui")
            selected, method = interactive_pocket_selection(sample_pockets)
            assert method == "graphical_gui"
            assert mock_launch.called

    # 'g' string input
    with patch("builtins.input", return_value="g"):
        with patch("core.pocket_gui.launch_pocket_gui") as mock_launch:
            mock_launch.return_value = (sample_pockets[0], "graphical_gui")
            selected, method = interactive_pocket_selection(sample_pockets)
            assert method == "graphical_gui"
            assert mock_launch.called


def test_prompt_pocket_confirmation_gui_option(sample_pockets):
    # 'g' in confirmation prompt
    with patch("builtins.input", return_value="g"):
        with patch("core.pocket_gui.launch_pocket_gui") as mock_launch:
            mock_launch.return_value = (sample_pockets[0], "graphical_gui")
            selected, method = prompt_pocket_confirmation_with_selection(sample_pockets)
            assert method == "graphical_gui"
            assert mock_launch.called


def test_display_selected_pocket_summary(sample_pockets, capsys):
    display_selected_pocket_summary(sample_pockets[0], "graphical_gui")
    captured = capsys.readouterr().out
    assert "Selected Docking Pocket" in captured
    assert "Graphical Interface" in captured
    assert "110.983" in captured
    assert "113.918" in captured
    assert "115.921" in captured
