"""
Convenience launcher for the Stock Price Predictor FastAPI application.
Automatically selects an available port if port 8000 is occupied or restricted.
"""

import socket
import webbrowser
import threading
import time
import uvicorn


def find_available_port(preferred_ports=(8000, 8080, 8050, 5000, 5050, 8001, 8888)):
    """Finds the first available port from preferred list or lets the OS assign one."""
    for port in preferred_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    # Fallback to any free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def open_browser(port: int):
    time.sleep(1.2)
    url = f"http://127.0.0.1:{port}"
    try:
        webbrowser.open(url)
    except Exception:
        pass


if __name__ == "__main__":
    port = find_available_port()
    
    print("=" * 60)
    print("  🚀 Stock Price Predictor Web Application")
    print(f"  🌐 App URL:  http://127.0.0.1:{port}")
    print(f"  📚 API Docs: http://127.0.0.1:{port}/docs")
    print("=" * 60)

    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    # Run uvicorn on the detected open port
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)
    except Exception as e:
        print(f"\n[Warning] Direct start failed ({e}), trying fallback port 8080...")
        uvicorn.run("main:app", host="127.0.0.1", port=8080, reload=False)
