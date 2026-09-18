"""Refuse occupied ports; never terminate an unrelated listener."""
import socket

def require_ports_free(ports):
    for port in ports:
        if not 1 <= int(port) <= 65535:
            raise ValueError(f"Invalid port: {port}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("", int(port)))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is occupied; choose another base port. No process was stopped.") from exc
