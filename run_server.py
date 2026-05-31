import subprocess
import sys
import time

# 启动服务器
process = subprocess.Popen(
    [sys.executable, 'start_server.py'],
    stdout=open('server_output.log', 'w'),
    stderr=subprocess.STDOUT,
    cwd='c:/Users/13043/Desktop/FindFakePlateVehicle-web'
)

print(f"服务器已启动，PID: {process.pid}")
print("等待服务器初始化...")
time.sleep(10)
print("请访问 http://localhost:8090")
