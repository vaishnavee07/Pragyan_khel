"""
Modes Package - Application-specific AI modes
Each mode combines detection, tracking, and business logic for specific use cases
"""
from .attendance_mode import AttendanceMode

__all__ = ['AttendanceMode']
