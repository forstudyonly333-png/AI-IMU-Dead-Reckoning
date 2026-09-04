import socket
import json
import time

class IMUReceiver:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.buffer = ""

    def start(self):
        """Initializes TCP server and blocks until client connects."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"📡 IMU Receiver listening on {self.host}:{self.port}...")
        self.client_socket, addr = self.server_socket.accept()
        print(f"✅ Device connected from {addr}")

    def receive(self):
        """Reads a single newline-delimited JSON packet."""
        while "\n" not in self.buffer:
            try:
                data = self.client_socket.recv(1024).decode("utf-8")
                if not data:
                    return None
                self.buffer += data
            except Exception:
                return None

        line, self.buffer = self.buffer.split("\n", 1)
        try:
            return json.loads(line.strip())
        except Exception:
            return {}

    def close(self):
        if self.client_socket:
            try:
                self.client_socket.close()
            except Exception:
                pass
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass