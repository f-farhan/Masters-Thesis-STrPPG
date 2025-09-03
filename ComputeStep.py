import os
import joblib

CACHE_DIR = "./resultats/cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

class Step:
    _registered_steps = []
    _recompute_flag = False
    
    def __init__(self, name, compute_func, recompute):
        """
        Represents a computation step.
        
        :param name: Unique name for the step.
        :param compute_func: Function to compute the step's result.
        """
        self.name         = name
        self.compute_func = compute_func
        self.recompute    = recompute
        self.cache_path   = os.path.join(CACHE_DIR, f"{self.name}.npy")
        self._registered_steps.append(self)

    def save_result(self, result):
        """Save intermediate result using NumPy's np.save."""
        # np.save(self.cache_path, result)
        joblib.dump(result, self.cache_path)

    def load_result(self):
        """Load intermediate result if it exists, otherwise return None."""
        # if os.path.exists(self.cache_path):
        #     return np.load(self.cache_path, allow_pickle=True)
        # return None
        if os.path.exists(self.cache_path):
            return joblib.load(self.cache_path)
        return None

    def run(self, *args, **kwargs):
        """
        Run the step: load cached result if available, otherwise compute and save.
        
        :param recompute: Whether to force recomputation.
        :return: Computed or loaded result.
        """
        if self.recompute:
            Step._recompute_flag = True
        
        if not self.recompute and not Step._recompute_flag:
            cached_result = self.load_result()
            if cached_result is not None:
                print(f"Loading {self.name} result from cache.")
                return cached_result
        print(f"Computing {self.name}...")
        result = self.compute_func(*args, **kwargs)
        self.save_result(result)
        return result
                              

                
