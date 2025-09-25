from src.confidence import lor

K_EASY = 2
K_MEDIUM = 4
K_HARD = 7


def compute_confidence(score, length, alpha=0.05):
    alpha = alpha / 2
    lower_bound = lor(score, length, alpha)
    upper_bound = 1 - lor(length - score, length, alpha)
    return lower_bound, upper_bound


def lcata_score():
    pass

def consistency_decay_scores():
    pass



