from __future__ import annotations

import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from skeleton_action.constants import J
from skeleton_action.inference.predictor import Predictor
from skeleton_action.inference.realtime import (
    NO_PERSON_STATUS,
    WARMING_UP,
    RealtimeConfig,
    RealtimeRecognizer,
    resample_window,
)
from skeleton_action.sensors.base import NO_PERSON, TRACKED, SkeletonFrame
from skeleton_action.sensors.kinect_v1_mapping import (
    POS_INFERRED,
    POS_NOT_TRACKED,
    POS_TRACKED,
    SDK,
    KinectV1Mapper,
    mirror_kard,
)
from skeleton_action.sensors.sources import BridgeSource, KardReplaySource, SyntheticSource
from skeleton_action.training.trainer import train


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory, tiny_collection) -> Path:
    tmp = tmp_path_factory.mktemp("inf")
    interim = tmp / "interim.npz"
    tiny_collection.save(interim)
    cfg = {
        "name": "inf",
        "dataset": {
            "kind": "realworld",
            "interim_path": str(interim),
            "processed_dir": None,
            "preprocess": {"num_frames": 16},
            "augment": {"enabled": False},
        },
        "split": {"train": [1, 2], "val": [3], "test": [4]},
        "model": {
            "name": "stgcn",
            "params": {"channels": [8, 8], "strides": [1, 1], "temporal_kernel": 3},
        },
        "training": {"seed": 0, "device": "cpu", "epochs": 1, "batch_size": 32},
    }
    return Path(train(cfg, tmp / "out", evaluate_test=False)["checkpoint"])


def test_offline_inference_without_kinect(ckpt: Path, standing_seq) -> None:
    # The offline path must not load any Kinect / OpenCV module.
    import subprocess

    code = (
        "import sys; sys.path.insert(0, 'src'); "
        "import skeleton_action.inference.predictor; "
        "bad = [m for m in sys.modules if 'kinect_sdk' in m or m == 'cv2']; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    p = Predictor(ckpt)
    r = p.predict_raw(standing_seq, topk=3)
    assert r.valid and r.label in p.class_names and len(r.topk) == 3
    assert abs(sum(r.probs) - 1) < 1e-5 and r.forward_ms > 0
    synth = np.stack([f.joints for f in SyntheticSource(num_frames=40)])
    assert p.predict_raw(synth).valid
    feats = np.zeros(p.input_shape, np.float32)
    assert p.predict_features(feats).valid
    assert not p.predict_features(np.zeros((3, 15, 3), np.float32)).valid


def test_invalid_inputs_do_not_predict(ckpt: Path, standing_seq) -> None:
    p = Predictor(ckpt)
    r = p.predict_raw(standing_seq[:5])
    assert not r.valid and r.label is None and "frames" in r.reason
    bad = standing_seq.copy()
    bad[:, J.TORSO] = 0.0  # torso never tracked
    r = p.predict_raw(bad)
    assert not r.valid and r.label is None
    assert not p.predict_raw(np.zeros((30, 14, 3), np.float32)).valid


def test_realtime_pipeline(ckpt: Path) -> None:
    p = Predictor(ckpt)
    rec = RealtimeRecognizer(
        p,
        RealtimeConfig(
            window_seconds=[1.0], stride_frames=3, confidence_threshold=0.0, margin_threshold=0.0
        ),
    )
    src = SyntheticSource(num_frames=90, fps=30.0)
    outs = [rec.update(f) for f in src]
    assert outs[0].status == WARMING_UP and outs[0].label is None
    assert any(o.updated and o.label is not None for o in outs[40:])
    # Tracking loss beyond the timeout resets the buffer and the decision.
    last_t = 90 / 30.0
    o = rec.update(SkeletonFrame(last_t + 0.1, None, None, NO_PERSON, "test"))
    assert o.status != NO_PERSON_STATUS  # short interruption: keep last state
    o = rec.update(SkeletonFrame(last_t + 1.0, None, None, NO_PERSON, "test"))
    assert o.status == NO_PERSON_STATUS and o.label is None and len(rec.buf) == 0


def test_realtime_thresholds_block_low_confidence(ckpt: Path) -> None:
    p = Predictor(ckpt)
    rec = RealtimeRecognizer(
        p, RealtimeConfig(window_seconds=[1.0], stride_frames=1, confidence_threshold=1.01)
    )
    outs = [rec.update(f) for f in SyntheticSource(num_frames=60)]
    assert all(o.label is None for o in outs)
    assert any(o.status == "uncertain" for o in outs)


def test_resample_window_with_missing_joint() -> None:
    times = np.arange(31) / 30.0
    joints = np.zeros((31, 15, 3), np.float32)
    joints[:, :, 0] = (times * 300)[:, None]
    valid = np.ones((31, 15), bool)
    valid[:10, J.L_HAND] = False
    seq, mask = resample_window(times, joints, valid, 1.0, 1.0, 15.0)
    assert seq.shape == (16, 15, 3) and mask.shape == (16, 15)
    np.testing.assert_allclose(seq[:, J.HEAD, 0], np.linspace(0, 300, 16), atol=1e-3)
    assert not mask[0, J.L_HAND] and mask[-1, J.L_HAND]


def test_kard_replay_source(tiny_collection, tmp_path: Path) -> None:
    path = tmp_path / "c.npz"
    tiny_collection.save(path)
    src = KardReplaySource.from_collection(path, subjects=[4], limit=2, gap_frames=3)
    frames = list(src)
    tracked = [f for f in frames if f.tracking == TRACKED]
    assert len(tracked) == int(
        tiny_collection.lengths[
            [tiny_collection.index_of(s) for s in ("a01_s04_e01", "a01_s04_e02")]
        ].sum()
    )
    assert tracked[0].meta["seq_id"] == "a01_s04_e01" and tracked[0].joints.shape == (15, 3)


# ----------------------------------------------------------------------- Kinect mapping


def sdk_skeleton() -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((20, 3))
    for i in range(20):
        pos[i] = [0.01 * i, 0.02 * i, 2.0 + 0.001 * i]
    pos[SDK["ShoulderRight"]] = [0.2, 1.3, 2.5]  # user's right = sensor's left = +X (SDK)
    pos[SDK["ShoulderLeft"]] = [-0.2, 1.3, 2.5]
    states = np.full(20, POS_TRACKED)
    return pos, states


def test_kinect_mapping_conversion() -> None:
    pos, states = sdk_skeleton()
    joints, valid, _ = KinectV1Mapper().convert(pos, states)
    assert joints.shape == (15, 3) and valid.all()
    np.testing.assert_allclose(
        joints[J.HEAD],
        [-1000 * pos[SDK["Head"], 0], 1000 * pos[SDK["Head"], 1], 1000 * pos[SDK["Head"], 2]],
        rtol=1e-6,
    )
    # KARD convention: right shoulder has the smaller X.
    assert joints[J.R_SHOULDER, 0] < joints[J.L_SHOULDER, 0]
    np.testing.assert_allclose(
        joints[J.R_FOOT],
        [
            -1000 * pos[SDK["AnkleRight"], 0],
            1000 * pos[SDK["AnkleRight"], 1],
            1000 * pos[SDK["AnkleRight"], 2],
        ],
        rtol=1e-6,
    )
    j2, _, _ = KinectV1Mapper(torso="spine_shoulder_mid").convert(pos, states)
    exp = (pos[SDK["Spine"]] + pos[SDK["ShoulderCenter"]]) / 2 * 1000
    np.testing.assert_allclose(j2[J.TORSO], [-exp[0], exp[1], exp[2]], rtol=1e-6)


def test_kinect_missing_and_inferred() -> None:
    pos, states = sdk_skeleton()
    states[SDK["HandLeft"]] = POS_NOT_TRACKED
    states[SDK["KneeRight"]] = POS_INFERRED
    j, v, info = KinectV1Mapper().convert(pos, states)
    assert not v[J.L_HAND] and np.all(j[J.L_HAND] == 0) and v[J.R_KNEE]
    assert info["inferred_mapped"] == 1
    _, v2, _ = KinectV1Mapper(accept_inferred=False).convert(pos, states)
    assert not v2[J.R_KNEE]


def test_mirror_swaps_sides() -> None:
    pos, states = sdk_skeleton()
    j, v, _ = KinectV1Mapper().convert(pos, states)
    m, _mv = mirror_kard(j, v)
    np.testing.assert_allclose(m[J.L_SHOULDER], j[J.R_SHOULDER] * [-1, 1, 1])
    np.testing.assert_allclose(m[J.HEAD], j[J.HEAD] * [-1, 1, 1])


def test_sdk_struct_layout_importable_anywhere() -> None:
    from skeleton_action.sensors import kinect_sdk_ctypes as k

    assert k.ctypes.sizeof(k.NUI_SKELETON_FRAME) == 2664
    if sys.platform != "win32":
        with pytest.raises(k.KinectError):
            k.KinectV1SdkReader()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_bridge_source_roundtrip() -> None:
    port = _free_port()
    src = BridgeSource(host="127.0.0.1", port=port, timeout=0.2, verbose=False)
    src.open()
    pos, states = sdk_skeleton()

    def client() -> None:
        time.sleep(0.1)
        with socket.create_connection(("127.0.0.1", port)) as c:
            c.sendall(b'{"type": "hello", "protocol": 1}\n')
            msg = {
                "type": "frame",
                "frame_number": 7,
                "sdk_timestamp_ms": 1000,
                "skeletons": [
                    {
                        "tracking_id": 3,
                        "state": 2,
                        "position": [0, 0, 2.5],
                        "joints": pos.tolist(),
                        "joint_states": states.tolist(),
                        "quality_flags": 0,
                    }
                ],
            }
            c.sendall((json.dumps(msg) + "\n").encode())
            c.sendall(
                (
                    json.dumps(
                        {
                            "type": "frame",
                            "frame_number": 8,
                            "sdk_timestamp_ms": 1033,
                            "skeletons": [],
                        }
                    )
                    + "\n"
                ).encode()
            )
            time.sleep(0.3)

    th = threading.Thread(target=client)
    th.start()
    got = []
    deadline = time.time() + 5
    while time.time() < deadline and len([f for f in got if f.tracking != "lost"]) < 2:
        got.append(src.read())
    th.join()
    src.close()
    real = [f for f in got if f.tracking != "lost"]
    assert real, "no frame received over the bridge"
    tracked = [f for f in real if f.tracking == TRACKED]
    if tracked:  # the receiver may drain to the newest frame if both arrived together
        f = tracked[0]
        assert f.frame_number == 7 and f.meta["tracking_id"] == 3 and f.valid.all()
        assert f.joints[J.R_SHOULDER, 0] < f.joints[J.L_SHOULDER, 0]
    assert real[-1].frame_number == 8 and real[-1].tracking == NO_PERSON
