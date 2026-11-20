# Shared helpers to load the serialized facial encodings.
import pickle


# [ADDED] Compatibility with trusted encodings created using NumPy 2
# when running NumPy 1.x (numpy._core -> numpy.core).
class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_encodings(path):
    """Load a trusted encodings pickle using CompatibleUnpickler."""
    with open(path, "rb") as f:
        return CompatibleUnpickler(f).load()
