import os
import pty
import select
import struct
import fcntl
import termios
import subprocess
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from sqlalchemy.orm import Session
from database import get_db, VirtualEnvironment
import json

router = APIRouter()


class TerminalSession:
    """Manages a PTY session for a virtual environment or regular shell"""

    def __init__(self, venv_path: str = None, venv_name: str = None):
        self.venv_path = venv_path
        self.venv_name = venv_name
        self.fd = None
        self.pid = None
        self.shell_process = None

    def start(self):
        """Start a PTY session with bash, optionally inside a virtual environment"""
        # Create a pseudo-terminal
        self.pid, self.fd = pty.fork()

        if self.pid == 0:  # Child process
            # Set up environment
            env = os.environ.copy()

            if self.venv_path:
                # Activate virtual environment by setting PATH and VIRTUAL_ENV
                venv_bin = os.path.join(self.venv_path, "bin")
                env["VIRTUAL_ENV"] = self.venv_path
                env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

                # Remove PYTHONHOME if set (can interfere with venv)
                env.pop("PYTHONHOME", None)

                # Set PS1 to show venv name
                env["PS1"] = f"({self.venv_name}) \\u@\\h:\\w$ "
            else:
                # Regular shell without venv
                env["PS1"] = "\\u@\\h:\\w$ "

            # Execute bash with the updated environment
            os.execvpe("/bin/bash", ["/bin/bash"], env)

        # Parent process - set fd to non-blocking
        flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def resize(self, cols: int, rows: int):
        """Resize the terminal"""
        if self.fd is not None:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, winsize)

    def write(self, data: str):
        """Write data to the PTY"""
        if self.fd is not None:
            os.write(self.fd, data.encode())

    def read(self, timeout=0.1):
        """Read data from the PTY with timeout"""
        if self.fd is None:
            return None

        # Use select to check if data is available
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if ready:
            try:
                data = os.read(self.fd, 10240)
                return data.decode('utf-8', errors='replace')
            except OSError:
                return None
        return ""

    def close(self):
        """Close the PTY session"""
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

        if self.pid is not None:
            try:
                os.kill(self.pid, 9)
                os.waitpid(self.pid, 0)
            except (OSError, ChildProcessError):
                pass
            self.pid = None


@router.websocket("/ws")
async def terminal_websocket(
    websocket: WebSocket,
    venv_id: int = None,
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for terminal session"""

    venv = None
    venv_path = None
    venv_name = None

    # If venv_id is provided, get the virtual environment
    if venv_id is not None:
        venv = db.query(VirtualEnvironment).filter(VirtualEnvironment.id == venv_id).first()
        if not venv:
            await websocket.close(code=1008, reason="Virtual environment not found")
            return

        # Verify venv path exists
        if not os.path.exists(venv.path):
            await websocket.close(code=1008, reason="Virtual environment path does not exist")
            return

        venv_path = venv.path
        venv_name = venv.name

    await websocket.accept()

    # Create terminal session (with or without venv)
    terminal = TerminalSession(venv_path, venv_name)

    try:
        # Start the PTY
        terminal.start()

        # Send initial message
        if venv:
            await websocket.send_json({
                "type": "output",
                "data": f"\r\nConnected to virtual environment: {venv.name}\r\n"
            })
        else:
            await websocket.send_json({
                "type": "output",
                "data": f"\r\nConnected to system shell\r\n"
            })

        # Create tasks for reading from PTY and WebSocket
        async def read_from_pty():
            """Read from PTY and send to WebSocket"""
            while True:
                try:
                    # Read from PTY (non-blocking)
                    output = await asyncio.get_event_loop().run_in_executor(
                        None, terminal.read, 0.05
                    )

                    if output:
                        await websocket.send_json({
                            "type": "output",
                            "data": output
                        })

                    await asyncio.sleep(0.01)  # Small delay to prevent busy-waiting

                except Exception as e:
                    print(f"Error reading from PTY: {e}")
                    break

        async def read_from_websocket():
            """Read from WebSocket and write to PTY"""
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if message["type"] == "input":
                        # Write input to PTY
                        await asyncio.get_event_loop().run_in_executor(
                            None, terminal.write, message["data"]
                        )

                    elif message["type"] == "resize":
                        # Resize terminal
                        cols = message.get("cols", 80)
                        rows = message.get("rows", 30)
                        await asyncio.get_event_loop().run_in_executor(
                            None, terminal.resize, cols, rows
                        )

                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error reading from WebSocket: {e}")
                    break

        # Run both tasks concurrently
        await asyncio.gather(
            read_from_pty(),
            read_from_websocket(),
            return_exceptions=True
        )

    except Exception as e:
        print(f"Terminal session error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

    finally:
        # Clean up
        terminal.close()
        try:
            await websocket.close()
        except:
            pass


@router.get("/test")
async def test_terminal():
    """Test endpoint to verify terminal API is loaded"""
    return {"status": "Terminal API is running"}
