"""dronekit: shared data loading, inference engine, scorer and plots for the drone-QA project.

One implementation of the evaluation protocol, imported by every notebook.
"""
__version__ = "0.1.0"

VISDRONE_CLASSES = ["pedestrian", "people", "bicycle", "car", "van",
                    "truck", "tricycle", "awning-tricycle", "bus", "motor"]
