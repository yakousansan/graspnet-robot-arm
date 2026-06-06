import json
import socket

import pytest


def test_build_pose_request_includes_protocol_fields():
    from graspnt_rm.udp_client import build_pose_request

    assert build_pose_request(seq=7) == {
        "version": 1,
        "type": "pose_request",
        "seq": 7,
    }


def test_build_grasp_execute_request_uses_only_grasp_plan():
    from graspnt_rm.udp_client import build_grasp_execute_request

    message = build_grasp_execute_request(
        seq=8,
        command_id="cmd-8",
        plan={
            "score": 0.82,
            "width": 0.045,
            "pre_grasp_pose": [0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
            "grasp_pose": [0.1, 0.2, 0.2, 0.0, 0.1, 0.2],
            "lift_pose": [0.1, 0.2, 0.4, 0.0, 0.1, 0.2],
        },
    )

    assert message == {
        "version": 1,
        "type": "grasp_execute",
        "seq": 8,
        "command_id": "cmd-8",
        "frame": "base",
        "unit": "m_rad",
        "pre_grasp_pose": [0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
        "grasp_pose": [0.1, 0.2, 0.2, 0.0, 0.1, 0.2],
        "lift_pose": [0.1, 0.2, 0.4, 0.0, 0.1, 0.2],
        "score": 0.82,
        "width": 0.045,
    }


def test_extract_current_end_pose_rejects_failed_response():
    from graspnt_rm.udp_client import extract_current_end_pose

    with pytest.raises(RuntimeError, match="pose_response failed"):
        extract_current_end_pose(
            {
                "type": "pose_response",
                "status": "failed",
                "reason": "robot disconnected",
            }
        )


def test_udp_robot_client_requests_pose_and_executes_grasp(monkeypatch):
    from graspnt_rm.udp_client import UdpRobotClient

    class FakeSocket:
        def __init__(self):
            self.sent = []
            self.timeouts = []
            self.responses = [
                {
                    "version": 1,
                    "type": "pose_response",
                    "seq": 1,
                    "status": "ok",
                    "end_pose": [0.4, -0.1, 0.2, 0.0, 0.0, 0.0],
                },
                {
                    "version": 1,
                    "type": "ack",
                    "seq": 2,
                    "command_id": "cmd-2",
                    "status": "accepted",
                },
                {
                    "version": 1,
                    "type": "result",
                    "seq": 2,
                    "command_id": "cmd-2",
                    "status": "success",
                },
            ]

        def settimeout(self, timeout):
            self.timeouts.append(timeout)

        def sendto(self, payload, address):
            self.sent.append((json.loads(payload.decode("utf-8")), address))
            return len(payload)

        def recvfrom(self, _size):
            if not self.responses:
                raise socket.timeout()
            response = self.responses.pop(0)
            return json.dumps(response).encode("utf-8"), ("127.0.0.1", 6556)

        def close(self):
            pass

    fake_socket = FakeSocket()
    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: fake_socket)

    client = UdpRobotClient(
        host="127.0.0.1",
        port=6556,
        ack_timeout_sec=0.1,
        result_timeout_sec=0.2,
        max_retries=1,
    )

    pose_response = client.request_pose()
    result = client.execute_grasp(
        plan={
            "score": 0.82,
            "width": 0.045,
            "pre_grasp_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            "grasp_pose": [0.1, 0.2, 0.2, 0.0, 0.0, 0.0],
            "lift_pose": [0.1, 0.2, 0.4, 0.0, 0.0, 0.0],
        },
        command_id="cmd-2",
    )

    assert pose_response["end_pose"] == [0.4, -0.1, 0.2, 0.0, 0.0, 0.0]
    assert result["status"] == "success"
    assert fake_socket.sent[0][0]["type"] == "pose_request"
    assert fake_socket.sent[0][1] == ("127.0.0.1", 6556)
    assert fake_socket.sent[1][0]["type"] == "grasp_execute"


def test_udp_robot_client_returns_cancelled_grasp_result(monkeypatch):
    from graspnt_rm.udp_client import UdpRobotClient

    class FakeSocket:
        def __init__(self):
            self.sent = []
            self.responses = [
                {
                    "version": 1,
                    "type": "ack",
                    "seq": 1,
                    "command_id": "cmd-1",
                    "status": "accepted",
                },
                {
                    "version": 1,
                    "type": "result",
                    "seq": 1,
                    "command_id": "cmd-1",
                    "status": "cancelled",
                    "reason": "operator declined",
                },
            ]

        def settimeout(self, timeout):
            pass

        def sendto(self, payload, address):
            self.sent.append((json.loads(payload.decode("utf-8")), address))
            return len(payload)

        def recvfrom(self, _size):
            response = self.responses.pop(0)
            return json.dumps(response).encode("utf-8"), ("127.0.0.1", 6556)

        def close(self):
            pass

    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: FakeSocket())
    client = UdpRobotClient("127.0.0.1", 6556, max_retries=1)

    result = client.execute_grasp(
        plan={
            "score": 0.82,
            "width": 0.045,
            "pre_grasp_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            "grasp_pose": [0.1, 0.2, 0.2, 0.0, 0.0, 0.0],
            "lift_pose": [0.1, 0.2, 0.4, 0.0, 0.0, 0.0],
        },
        command_id="cmd-1",
    )

    assert result["status"] == "cancelled"
    assert result["reason"] == "operator declined"
