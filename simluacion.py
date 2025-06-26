import subprocess
import os

def start_camera(id=1):
    # Pasar VIDEO_ID como variable de entorno
    env = os.environ.copy()
    env["VIDEO_ID"] = str(id)

    subprocess.run("python app.py", shell=True, env=env)

    return 0