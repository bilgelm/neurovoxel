
"""
DEPRECATED: NeuroVoxel UI has moved.
Please run the new Streamlit UI via app/main.py.
"""

import subprocess
import sys

def main():
    print("DEPRECATED: NeuroVoxel UI has moved. Please run the new Streamlit UI via app/main.py.")
    subprocess.run([sys.executable, "app/main.py"])

if __name__ == "__main__":
    main()
