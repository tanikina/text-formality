def normalize_score(x: float, old_min=-3, old_max=3, new_min=0, new_max=1):
    """Scales a value from the range [old_min,old_max] to [new_min,new_max]."""
    norm_score = (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return {"label": round(norm_score, 2)}
