"""
WCR WinML helper for ONNX Runtime GenAI execution provider registration
Adapted from WCR Tools/winml_helper.py for SpecFlow edge inference
"""
import os
import sys
import json
import subprocess
from pathlib import Path

# Path to the embedded winml worker
WINML_WORKER_FILE = str(Path(__file__).parent / 'wcr_winml_worker.py')

def _get_execution_provider_paths() -> dict[str, str]:
    """Get execution provider paths using WinML discovery"""
    print("\n=== Starting WCR execution provider discovery ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Current script path: {__file__}")

    print(f"\nExecuting WinML worker with Python interpreter: {sys.executable}")

    try:
        # Run subprocess with the embedded worker code
        process = subprocess.Popen(
            [sys.executable, WINML_WORKER_FILE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()

        print(f"\nSubprocess completed with return code: {process.returncode}")

        if stderr:
            print(f"\nSubprocess STDERR output:\n{'-' * 50}")
            print(stderr)
            print(f"{'-' * 50}")

        print(f"\nSubprocess STDOUT output:\n{'-' * 50}")
        print(stdout if stdout else "(no stdout output)")
        print(f"{'-' * 50}")

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, 
                [sys.executable, WINML_WORKER_FILE], 
                stdout, 
                stderr
            )

        # Find the last line that looks like JSON (in case there's extra output)
        json_output = None
        for line in stdout.strip().split('\n'):
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                json_output = line

        if not json_output:
            print("ERROR: No JSON output found in stdout")
            print("Attempting to parse entire stdout as JSON...")
            json_output = stdout

        print(f"\nParsing JSON output: {json_output[:100]}..." if len(json_output) > 100 else f"\nParsing JSON output: {json_output}")
        paths = json.loads(json_output)
        print(f"Parsed JSON successfully. Found {len(paths)} providers:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        print("=== Completed WCR execution provider discovery ===\n")
        return paths

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Subprocess failed with return code {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        print(f"\nERROR: Failed to parse JSON output")
        print(f"Error: {e}")
        print(f"Raw stdout was: {stdout}")
        print(f"Raw stderr was: {stderr}")
        raise
    except Exception as e:
        print(f"\nERROR: Unexpected error in _get_execution_provider_paths")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise

# Global flag to track registration status
_genai_ep_registered = False

def register_execution_providers_to_onnxruntime_genai():
    """Register WinML execution providers to ONNX Runtime GenAI"""
    import onnxruntime_genai as og
    global _genai_ep_registered
    
    if _genai_ep_registered:
        print("Execution providers already registered to onnxruntime_genai.")
        return
    
    try:
        execution_providers = _get_execution_provider_paths()
        for provider_name, provider_path in execution_providers.items():
            print(f"Registering {provider_name} with onnxruntime_genai from path: {provider_path}")
            og.register_execution_provider_library(provider_name, provider_path)
        
        _genai_ep_registered = True
        print(f"Successfully registered {len(execution_providers)} execution providers")
        
    except Exception as e:
        print(f"Failed to register execution providers: {e}")
        print("Make sure winui3 and onnxruntime-winml packages are installed")
        raise

def is_qnn_provider_available():
    """Check if QNN provider is available (simplified check)"""
    try:
        # Just try to register - if it works, QNN is available
        register_execution_providers_to_onnxruntime_genai()
        return True
    except Exception as e:
        print(f"QNN provider not available: {e}")
        return False