import subprocess
import os
import platform

def start_camera(id=1):
    env = os.environ.copy()
    env["VIDEO_ID"] = str(id)

    if platform.system() == "Windows":
        subprocess.Popen(f'start cmd /k "python app.py"', shell=True, env=env)
    else:
        # Para otros sistemas, como Linux o macOS
        subprocess.Popen(['x-terminal-emulator', '-e', 'python app.py'], env=env)

    return 0