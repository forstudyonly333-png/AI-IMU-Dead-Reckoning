import json
import socket
import time


HOST = "127.0.0.1"
PORT = 5000


def main():

    print("\n📱 Starting simulated smartphone...")
    print("--------------------------------")

    client = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    )

    client.connect(
        (HOST, PORT)
    )

    print("✅ Connected to receiver")

    start_time = time.time()

    try:

        for i in range(100):

            timestamp = time.time() - start_time

            packet = {
                "timestamp": timestamp,

                "ax": 0.0,
                "ay": 0.0,
                "az": 9.81,

                "gx": 0.0,
                "gy": 0.0,
                "gz": 0.0,

                "mx": 30.0,
                "my": 5.0,
                "mz": -40.0
            }

            message = (
                json.dumps(packet)
                + "\n"
            )

            client.sendall(
                message.encode("utf-8")
            )

            print(
                f"Sent sample {i + 1}: "
                f"t={timestamp:.2f}s"
            )

            time.sleep(0.05)

    finally:

        client.close()

        print(
            "\n📱 Simulated smartphone stopped."
        )


if __name__ == "__main__":
    main()