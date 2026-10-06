"""Network workarounds."""

from __future__ import annotations

import os
import socket


def force_ipv4_if_requested() -> bool:
    """Resolve host names to IPv4 only when ``RIM_FORCE_IPV4=1``.

    On networks that advertise IPv6 routes without carrying IPv6 traffic, Python
    clients try every IPv6 address in turn before falling back, which makes each
    request take minutes. The variable is opt-in and changes nothing by default.
    """
    if os.environ.get("RIM_FORCE_IPV4", "") != "1":
        return False
    original = socket.getaddrinfo
    if getattr(original, "_rim_ipv4_only", False):
        return True

    def ipv4_only(host, port, family=0, *args, **kwargs):
        return original(host, port, socket.AF_INET, *args, **kwargs)

    ipv4_only._rim_ipv4_only = True
    socket.getaddrinfo = ipv4_only
    return True
