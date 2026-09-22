"""The demo's runner and HTTP API, on a synthetic video (no network)."""

import http.client
import json
import shutil
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from directConv import edge
from directConv.demo import runner, server


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    frames = np.random.default_rng(0).integers(0, 256, (12, 288, 512, 3), dtype=np.uint8)
    iio.imwrite(path, frames, fps=10, codec="libx264", plugin="FFMPEG", macro_block_size=1)
    return str(path)


@pytest.fixture(scope="module")
def _measured_once(video, tmp_path_factory):
    out = tmp_path_factory.mktemp("media")
    return runner.run(video, 6, "640×360", str(out)), out


@pytest.fixture()
def measured(_measured_once, video):
    """The run the tests read. The runner keeps only the last run, so if another test replaced it, measure again."""
    result, out = _measured_once
    last = runner.last_result()
    if last is None or last["run_id"] != result["run_id"]:
        result = runner.run(video, 6, "640×360", str(out))
    return result, out


def test_run_measures_every_strategy_and_they_all_equal_numpy(measured):
    result, _ = measured
    assert result["max_abs_diff"] == 0.0
    assert {"stream@360x640", "heap@360x640", "stack@72x128", "heap@72x128"} <= set(result["measured"])
    assert all(m["ms_per_frame"] > 0 for m in result["measured"].values())


def test_run_reports_a_distribution_not_a_single_number(measured):
    for key, m in measured[0]["measured"].items():
        assert m["p5_ms"] <= m["ms_per_frame"] <= m["p95_ms"], key
        assert m["batches"] >= 2 and m["frames"] == measured[0]["frames"], key
    n = measured[0]["numpy"]
    assert n["p5_ms"] <= n["ms_per_frame"] <= n["p95_ms"] and n["batches"] >= 2


def test_a_video_longer_than_its_source_loops_forward_and_back(video, tmp_path):
    result = runner.run(video, 30, "640×360", str(tmp_path))  # the clip has 12 frames
    assert result["frames"] == 30 and result["source_frames"] == 12 and result["looped"]
    assert result["max_abs_diff"] == 0.0
    play = runner.play(result["run_id"], "stream", "stm32", 20, str(tmp_path))
    assert play["frames"] == 30 and len(play["processed_indices"]) == play["processed"]


def test_sequence_plays_forward_then_backward_without_a_cut():
    assert runner.sequence(4, 10).tolist() == [0, 1, 2, 3, 2, 1, 0, 1, 2, 3]
    assert runner.sequence(1, 3).tolist() == [0, 0, 0]
    seq = runner.sequence(280, 3000)
    assert (abs(np.diff(seq)) == 1).all()  # consecutive frames are always neighbours


def test_frame_budget_follows_the_cost_of_numpy():
    assert runner.max_frames("64×36") == runner.MAX_FRAMES
    assert runner.max_frames("1920×1080") < runner.max_frames("1280×720") < runner.max_frames("640×360")
    assert runner.max_frames("1280×720") > 500  # long enough for several seconds of video


def test_run_explains_where_numpy_loses_time(measured):
    b = measured[0]["numpy"]["breakdown"]
    assert b["view_is_zero_copy"] and b["flattened_is_a_copy"]  # sliding_window_view is a view, tensordot's reshape copies
    assert 0 < b["copy_share"] < 1 and b["copy_mb_per_frame"] == pytest.approx(runner.patch_bytes(360, 640) / 1e6, rel=0.01)


def test_numpy_needs_far_more_working_memory_than_streaming(measured):
    result, _ = measured
    assert result["numpy"]["memory"]["working_mb"] * 1e6 > 100 * result["cards"]["stm32"]["stream"]["bytes"]


def test_cards_carry_the_measurements(measured):
    cards = measured[0]["cards"]
    host = cards["host"]  # 640x360 is 2.7 MB of tensors: it fits an 8 MiB stack, so no reduced frame is needed
    assert host["reduced"]["unneeded"]
    assert host["stack"]["host_ms"] > 0 and host["heap"]["host_ms"] > 0 and host["stream"]["host_ms"] > 0
    assert cards["stm32"]["reduced"]["host_ms"] > 0 and cards["stm32"]["stream"]["chip_ms"] > 0


def test_heap_costs_no_measurable_time_against_the_stack(measured):
    hs = measured[0]["heap_vs_stack"]["stm32"]
    assert 0.3 < hs["ratio"] < 3  # same code, different storage: the same order of time, either way


def test_play_drops_frames_when_the_device_is_slower_than_the_camera(measured):
    result, out = measured
    slow = runner.play(result["run_id"], "stream", "stm32", 20, str(out))
    assert slow["dropped"] > 0 and slow["processed"] + slow["dropped"] == slow["frames"]
    assert slow["effective_fps"] < 20
    fast = runner.play(result["run_id"], "reduced", "stm32", 20, str(out))
    assert fast["dropped"] == 0 and fast["size"]["name"] == "128×72"
    assert all((Path(out) / Path(u).name).is_file() for u in slow["videos"].values())


def test_play_refuses_what_does_not_fit(measured):
    result, out = measured
    with pytest.raises(ValueError, match="allocator"):
        runner.play(result["run_id"], "heap", "stm32", 20, str(out))
    with pytest.raises(ValueError, match="gone"):
        runner.play("stale", "stream", "stm32", 20, str(out))


def test_play_rejects_a_camera_rate_beyond_the_cap(measured):
    result, out = measured
    with pytest.raises(ValueError, match="camera fps"):
        runner.play(result["run_id"], "stream", "stm32", runner.MAX_CAMERA_FPS + 1, str(out))


def test_the_chip_keeps_up_with_a_120_fps_camera_at_128x72_but_not_240(measured):
    result, out = measured
    # 128x72, direct: 7.4 ms per frame on the chip = 135 fps, between the two camera rates.
    at_120 = runner.play(result["run_id"], "reduced", "stm32", 120, str(out))
    assert at_120["size"]["name"] == "128×72" and at_120["dropped"] == 0
    at_240 = runner.play(result["run_id"], "reduced", "stm32", 240, str(out))
    assert at_240["camera_fps"] == 240 and not at_240["clamped"]
    assert at_240["dropped"] > 0 and at_240["effective_fps"] == pytest.approx(135, rel=0.05)


def test_validation():
    with pytest.raises(ValueError, match="frames"):
        runner.validate_run(0, "1280×720")
    with pytest.raises(ValueError, match="resolution"):
        runner.validate_run(1, "8k")
    assert runner.max_frames("1920×1080") < runner.max_frames("640×360")  # numpy's copy bounds the frame count


@pytest.fixture()
def api(video, tmp_path):
    server.Handler.presets = {"clip": video}
    server.Handler.media_dir = tmp_path
    server.Handler.jobs = server.Jobs()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _post(port, path, body):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as err:
        return err.code, json.load(err)


def _wait(port, job):
    for _ in range(600):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/job/{job}") as response:
            data = json.load(response)
        if data["state"] != "running":
            return data
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_api_config_and_preview_need_no_run(api):
    with urllib.request.urlopen(f"http://127.0.0.1:{api}/api/config") as r:
        config = json.load(r)
    assert [p["name"] for p in config["ladder"]] == [n for n, _, _ in edge.LADDER]
    assert config["devices"]["stm32"]["budget"] == 121_538 and config["devices"]["stm32"]["verified"] == 72_244
    with urllib.request.urlopen(f"http://127.0.0.1:{api}/api/preview?resolution=" + urllib.parse.quote("1280×720")) as r:
        preview = json.load(r)
    assert preview["stm32"]["stream"]["fits"] and not preview["stm32"]["stack"]["fits"]


def test_api_run_then_play_and_one_job_at_a_time(api):
    body = {"preset": "clip", "frames": 4, "resolution": "640×360"}
    status, data = _post(api, "/api/run", body)
    assert status == 202
    assert _post(api, "/api/run", body)[0] == 409  # another job is running
    done = _wait(api, data["job"])
    assert done["state"] == "done", done
    status, play = _post(api, "/api/play", {"run_id": done["result"]["run_id"], "strategy": "stream", "device": "stm32", "camera_fps": 20})
    assert status == 202
    assert _wait(api, play["job"])["result"]["dropped"] > 0


def test_the_four_pages_and_their_assets_are_served(api):
    for path in ("/", "/lab", "/why", "/method", "/static/style.css", "/static/common.js", "/static/lab.js", "/tradeoffs.json"):
        with urllib.request.urlopen(f"http://127.0.0.1:{api}{path}") as r:
            assert r.status == 200 and len(r.read()) > 100, path
    for bad in ("/static/../server.py", "/static/runner.py", "/static/nope.js", "/nope"):
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{api}{bad}")
        assert err.value.code == 404, bad


def test_last_run_is_available_to_the_other_pages(api):
    status, data = _post(api, "/api/run", {"preset": "clip", "frames": 6, "resolution": "640×360"})
    assert status == 202 and _wait(api, data["job"])["state"] == "done"
    with urllib.request.urlopen(f"http://127.0.0.1:{api}/api/last") as r:
        last = json.load(r)["run"]
    assert last["frames"] == 6 and "breakdown" in last["numpy"]


def test_an_ip_address_as_host_is_refused_unless_lan_is_on(api):
    def status(host):
        conn = http.client.HTTPConnection("127.0.0.1", api)
        conn.request("GET", "/api/config", headers={"Host": host})
        return conn.getresponse().status

    assert status("192.168.1.27:8000") == 403 and status("localhost:8000") == 200
    server.Handler.lan = True
    try:
        assert status("192.168.1.27:8000") == 200
        assert status("evil.example") == 403 and status("192.168.1.27.evil.example") == 403  # names stay refused
    finally:
        server.Handler.lan = False


@pytest.mark.skipif(shutil.which("openssl") is None, reason="needs the openssl command")
def test_https_with_a_self_signed_certificate(tmp_path):
    server.Handler.media_dir = tmp_path
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    server.wrap_tls(httpd, "127.0.0.1", str(tmp_path))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        untrusted = ssl.create_default_context()
        untrusted.check_hostname, untrusted.verify_mode = False, ssl.CERT_NONE  # the certificate is self-signed
        with urllib.request.urlopen(f"https://127.0.0.1:{httpd.server_address[1]}/lab", context=untrusted) as r:
            assert r.status == 200 and b"Lab" in r.read()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_api_rejects_bad_requests(api):
    good = {"preset": "clip", "frames": 3, "resolution": "640×360"}
    assert _post(api, "/api/run", {**good, "preset": None, "url": "/etc/passwd"})[0] == 400
    assert _post(api, "/api/run", {**good, "preset": "nope"})[0] == 400
    assert _post(api, "/api/run", {**good, "frames": runner.MAX_FRAMES + 1})[0] == 400
    assert _post(api, "/api/run", {**good, "resolution": "8k"})[0] == 400
    conn = http.client.HTTPConnection("127.0.0.1", api)
    conn.request("GET", "/api/config", headers={"Host": "evil.example"})
    assert conn.getresponse().status == 403


def test_camera_rate_is_bounded_by_what_h264_can_carry():
    assert runner.max_camera_fps(70, 126) == runner.MAX_CAMERA_FPS == 240  # tiny frames: the 240 fps cap
    assert runner.max_camera_fps(1078, 1918) == 120  # 1080p: 8 160 macroblocks x 120 = level 5.1's limit
    assert runner.max_camera_fps(718, 1278) >= 240
