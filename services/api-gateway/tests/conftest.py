"""
Test configuration and fixtures.

Sets up Python path to allow importing from shared module.
"""

import sys
from pathlib import Path

# Add shared services directory to Python path
services_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(services_dir))
