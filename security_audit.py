import subprocess
import sys
from pathlib import Path

def run_security_audit():
    """
    Executes pip-audit on requirements.txt to scan all project dependencies
    for known vulnerabilities (CVEs).
    """
    project_root = Path(__file__).resolve().parent
    req_file = project_root / "requirements.txt"

    print("=" * 75)
    print(" RUNNING AUTOMATED DEPENDENCY VULNERABILITY AUDIT (pip-audit)")
    print("=" * 75)
    print(f"\nScanning dependencies in: {req_file.name}...\n")

    cmd = [sys.executable, "-m", "pip_audit", "-r", str(req_file)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout)

        if result.stderr:
            print("Audit Logs:\n", result.stderr)

        if result.returncode == 0:
            print("=" * 75)
            print(" SUCCESS: ZERO KNOWN VULNERABILITIES FOUND IN PROJECT DEPENDENCIES!")
            print("=" * 75)
            sys.exit(0)
        else:
            print("=" * 75)
            print(f" WARNING: SECURITY AUDIT DETECTED VULNERABILITIES (Exit Code: {result.returncode})")
            print("=" * 75)
            sys.exit(result.returncode)

    except FileNotFoundError:
        print("ERROR: pip-audit package not found! Install using 'pip install pip-audit'.")
        sys.exit(1)

if __name__ == "__main__":
    run_security_audit()
