"""
WCR WinML worker for execution provider discovery
Adapted from WCR Tools/winml_worker.py for SpecFlow edge inference
"""
import json
import sys
import traceback

# Debug flag - set to True to enable debug output
DEBUG_ENABLED = False

def debug_print(message, file=sys.stderr):
    """Print debug messages only if DEBUG_ENABLED is True"""
    if DEBUG_ENABLED:
        print(message, file=file)

# Load the onnxruntime.dll from the onnxruntime package to hack the
# WinML wrong onnxruntime.dll problem.
dll_handle = None

def winml_onnxruntime_dll_hack():
    """Load correct onnxruntime.dll for WinML compatibility"""
    global dll_handle
    if dll_handle is not None:
        return
    
    try:
        from importlib import metadata
        import ctypes
        from pathlib import Path
        
        ort_dist = metadata.distribution('onnxruntime-winml')
        ort_location = Path(str(ort_dist.locate_file('.'))).resolve()
        dll_path = ort_location / "onnxruntime" / 'capi' / 'onnxruntime.dll'
        
        if dll_path.exists():
            dll_handle = ctypes.CDLL(str(dll_path))
            debug_print(f"DEBUG: Loaded onnxruntime.dll from {dll_path}")
        else:
            debug_print(f"DEBUG: onnxruntime.dll not found at {dll_path}")
            
    except Exception as e:
        debug_print(f"DEBUG: Failed to load onnxruntime.dll hack: {e}")

def _get_ep_paths() -> dict[str, str]:
    """Get execution provider paths using WinML discovery"""
    debug_print("DEBUG: Entering _get_ep_paths() function")
    
    try:
        debug_print("DEBUG: Importing winui3 modules...")
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions,
            initialize
        )
        import winui3.microsoft.windows.ai.machinelearning as winml
        debug_print("DEBUG: Import successful")
    except ImportError as e:
        debug_print(f"DEBUG: ImportError: {e}")
        debug_print("DEBUG: winui3 package may not be installed")
        debug_print(f"DEBUG: sys.path = {sys.path}")
        raise
    except Exception as e:
        debug_print(f"DEBUG: Import failed: {type(e).__name__}: {e}")
        debug_print(f"DEBUG: Traceback:\\n{traceback.format_exc()}")
        raise
    
    eps = {}
    
    debug_print("DEBUG: Initializing WinUI3 with InitializeOptions.ON_NO_MATCH_SHOW_UI")
    try:
        with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
            debug_print("DEBUG: WinUI3 initialized successfully")
            
            debug_print("DEBUG: Getting default ExecutionProviderCatalog...")
            catalog = winml.ExecutionProviderCatalog.get_default()
            debug_print(f"DEBUG: Catalog obtained: {catalog}")
            
            debug_print("DEBUG: Finding all providers...")
            providers = catalog.find_all_providers()
            debug_print(f"DEBUG: Found {len(providers)} providers")
            
            for i, provider in enumerate(providers):
                debug_print(f"\\nDEBUG: Processing provider {i+1}/{len(providers)}")
                debug_print(f"DEBUG:   Provider object: {provider}")
                debug_print(f"DEBUG:   Provider name: {provider.name}")
                debug_print(f"DEBUG:   Provider library_path: {provider.library_path}")
                
                debug_print(f"DEBUG:   Calling ensure_ready_async()...")
                try:
                    provider.ensure_ready_async().get()
                    debug_print(f"DEBUG:   Provider {provider.name} is ready")
                except Exception as e:
                    debug_print(f"DEBUG:   ERROR ensuring provider ready: {type(e).__name__}: {e}")
                    continue
                    
                if provider.library_path == "":
                    debug_print(f"DEBUG:   Skipping provider {provider.name} with empty library_path")
                    continue
                    
                eps[provider.name] = provider.library_path
                debug_print(f"DEBUG:   Added to results: {provider.name} -> {provider.library_path}")
            
            debug_print(f"\\nDEBUG: Completed provider discovery. Total providers: {len(eps)}")
            
    except Exception as e:
        debug_print(f"DEBUG: Error during provider discovery: {type(e).__name__}: {e}")
        debug_print(f"DEBUG: Traceback:\\n{traceback.format_exc()}")
        raise
    
    debug_print("DEBUG: Returning execution provider paths")
    return eps

if __name__ == "__main__":
    debug_print("DEBUG: wcr_winml_worker.py running in __main__ block")
    try:
        winml_onnxruntime_dll_hack()
        eps = _get_ep_paths()
        debug_print(f"DEBUG: Got {len(eps)} execution providers")
        debug_print(f"DEBUG: Final JSON output: {json.dumps(eps)}")
        
        # This is the actual output that will be captured
        print(json.dumps(eps))
        sys.exit(0)
        
    except Exception as e:
        debug_print(f"DEBUG: Fatal error in main: {type(e).__name__}: {e}")
        debug_print(f"DEBUG: Traceback:\\n{traceback.format_exc()}")
        
        # Return empty dict on error so the process doesn't completely fail
        print(json.dumps({}))
        sys.exit(1)