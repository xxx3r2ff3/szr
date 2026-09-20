import sys
import os

if getattr(sys, "frozen", False):
    exe_dir = os.path.dirname(sys.executable)
else:
    exe_dir = os.path.dirname(os.path.abspath(__file__))

if exe_dir not in sys.path:
    sys.path.insert(0, exe_dir)

try:
    from modules.core.launcher import main
except ImportError:
    print(f"Error: Cannot find launcher module")
    print("Please ensure launcher.pyd exists in the core directory")
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
