import numpy as np

from stable_baselines3.common.base_class import Schedule


def linear_schedule(initial_value: float, final_value: float) -> Schedule:
    diff = initial_value - final_value
    def schedule_fn(progress_remaining: float) -> float:
        val = diff * progress_remaining + final_value
        return val
    
    return schedule_fn
        

def exponential_schedule(initial_value: float, final_value: float) -> Schedule:
    log_schedule = linear_schedule(np.log(initial_value), np.log(final_value))
    def schedule_fn(progress_remaining: float) -> float:
        val = np.exp(log_schedule(progress_remaining))
        return val

    return schedule_fn