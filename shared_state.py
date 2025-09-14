"""
Shared state manager for fall detection system.
This ensures all modules share the same global variables.
"""
import threading
from collections import deque

# Global fall detection results storage
g_images = {}
g_images_lock = threading.Lock()

# Global person classifier for fall verification
g_person_classifier = None
g_person_class_index = None

def get_images_dict():
    """Get the shared images dictionary"""
    return g_images

def get_images_lock():
    """Get the shared images lock"""
    return g_images_lock

def set_person_classifier(classifier, class_index):
    """Set the shared person classifier"""
    global g_person_classifier, g_person_class_index
    g_person_classifier = classifier
    g_person_class_index = class_index

def get_person_classifier():
    """Get the shared person classifier"""
    return g_person_classifier, g_person_class_index
