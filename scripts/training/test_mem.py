import psutil

for part in psutil.disk_partitions():
    try:
        usage = psutil.disk_usage(part.mountpoint)
        print(f"Drive {part.mountpoint} | Total: {usage.total / (1024**3):.1f} GB | Used: {usage.used / (1024**3):.1f} GB | Free: {usage.free / (1024**3):.1f} GB | Percent: {usage.percent}%")
    except Exception as e:
        print(f"Drive {part.mountpoint} could not be checked: {e}")
