import json
import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from src.confidence import lor

K_EASY = 2
K_MEDIUM = 4
K_HARD = 7


def compute_confidence(score, length, alpha=0.05):
    alpha = alpha / 2
    lower_bound = lor(score, length, alpha)
    upper_bound = 1 - lor(length - score, length, alpha)
    return lower_bound, upper_bound

def lcata_score(scores, k):
    lcata_k = {"easy": {"score": 0}, "medium": {"score": 0}, "hard": {"score": 0}}
    for i in k:
        for score in scores:
            if all(score[:i-1]):
                if i == K_EASY:
                    lcata_k["easy"]["score"] += 1
                elif i == K_MEDIUM:
                    lcata_k["medium"]["score"] += 1
                elif i == K_HARD:
                    lcata_k["hard"]["score"] += 1

    lcata_k["easy"]["score"] /= len(scores)
    lcata_k["medium"]["score"] /= len(scores)
    lcata_k["hard"]["score"] /= len(scores)

    confidence_lower, confidence_upper = compute_confidence(lcata_k["easy"], len(scores))
    lcata_k["easy"]["confidence"] = {"lower_bound": confidence_lower, "upper_bound": confidence_upper}
    confidence_lower, confidence_upper = compute_confidence(lcata_k["medium"], len(scores))
    lcata_k["medium"]["confidence"] = {"lower_bound": confidence_lower, "upper_bound": confidence_upper}
    confidence_lower, confidence_upper = compute_confidence(lcata_k["hard"], len(scores))
    lcata_k["hard"]["confidence"] = {"lower_bound": confidence_lower, "upper_bound": confidence_upper}

    return lcata_k

def combine_scores(all_results, tasks):
    data_frame_results = {
        "task": [],
        "model": [],
        "k": [],
        "score": []
    }

    for task in tasks:
        for model, results in all_results.items():
            print(results)
            for k, score in results[task].items():
                data_frame_results["task"].append(task)
                data_frame_results["model"].append(model)
                data_frame_results["k"].append(k)
                data_frame_results["score"].append(score["score"])

    return pd.DataFrame(data_frame_results)

    

def consistency_decay_scores(task, data_frame_results):
    # Compute consistency decay scores for a specific task
    task_data = data_frame_results[data_frame_results["task"] == task]
    sns.lineplot(data=task_data, x="k", y="score", hue="model")
    plt.xlabel("Number of Edits (k)")
    plt.ylabel("LCATA Score")
    plt.title(f"Consistency Decay for {task}")
    plt.savefig(f"consistency_decay_{task}.png")
    plt.clf()  # Clear the figure for the next plot
    # return task_data


def compute_lcata_k(folder_path, k):
    results = {}
    for provider in os.listdir(folder_path):
        provider_path = os.path.join(folder_path, provider)
        if not os.path.isdir(provider_path):
            continue
        for model in os.listdir(provider_path):
            results[model] = {}
            model_path = os.path.join(folder_path, model)
            if not os.path.isdir(model_path):
                continue
            for task in os.listdir(model_path):
                task_path = os.path.join(model_path, task)
                if not os.path.isdir(task_path):
                    continue
                if os.path.isdir(task_path):
                    all_scores = []
                    for file_path in os.listdir(task_path):
                        if file_path.endswith('.json'):
                            with open(os.path.join(task_path, file_path), 'r') as f:
                                    data = json.load(f)
                            scores = []
                            for entry in data["predictions"]:
                                if not entry["is_demonstration"]:
                                    scores.append(entry["correct"])
                            all_scores.append(scores)

                    lcata_k = lcata_score(all_scores, k)
                    results[model][task] = lcata_k
        return results


if __name__ == "__main__":
    folder_path = "detailed_models_results/moonshotai/"
    k = [2, 4, 7]
    results = compute_lcata_k(folder_path, k)
    print(json.dumps(results, indent=4))



