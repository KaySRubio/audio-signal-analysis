"""
Configuration pulls from environment variables, and if not set, uses these defaults
"""

import os
from typing import Optional

# ===================================================================
# AWS Configuration Example
# ===================================================================
AWS_REGION: str = os.getenv('AWS_REGION', 'us-east-1')

