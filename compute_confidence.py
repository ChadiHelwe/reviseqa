from src.confidence import lor
import os
import json


def analyse(data_path):
    data = json.load(data_path)
    alpha = 0.05/2
    length_data = data["metadata"]["dataset_length"]
    print(data["timestamp"])
    print("model", data["metadata"]["model_name"])
    for result in data["length_by_difficulty"]:
        if "implicit_no_reasoning_no_correction" == result:
            print("Standard + Implicit + No Feedback")
            print("Easy", data["length_by_difficulty"][result]["easy"]/length_data, lor(data["length_by_difficulty"][result]["easy"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["easy"], length_data, alpha))
            print("Medium", data["length_by_difficulty"][result]["medium"]/length_data, lor(data["length_by_difficulty"][result]["medium"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["medium"], length_data, alpha))
            print("Hard", data["length_by_difficulty"][result]["hard"]/length_data, lor(data["length_by_difficulty"][result]["hard"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["hard"], length_data, alpha))
            print("---"*10)

        elif "explicit_no_reasoning_no_correction" == result:
            print("Standard + Explicit + No Feedback")
            print("Easy", data["length_by_difficulty"][result]["easy"]/length_data, lor(data["length_by_difficulty"][result]["easy"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["easy"], length_data, alpha))
            print("Medium", data["length_by_difficulty"][result]["medium"]/length_data, lor(data["length_by_difficulty"][result]["medium"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["medium"], length_data, alpha))
            print("Hard", data["length_by_difficulty"][result]["hard"]/length_data, lor(data["length_by_difficulty"][result]["hard"], length_data, alpha), 1 - lor(length_data -data["length_by_difficulty"][result]["hard"], length_data, alpha))
            print("---"*10)

        
        elif "implicit_no_reasoning" in result:
            print("Standard + Implicit + Feedback")
            print("Easy", data["length_by_difficulty"][result]["easy"]/length_data, lor(data["length_by_difficulty"][result]["easy"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["easy"], length_data, alpha))
            print("Medium", data["length_by_difficulty"][result]["medium"]/length_data, lor(data["length_by_difficulty"][result]["medium"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["medium"], length_data, alpha))
            print("Hard", data["length_by_difficulty"][result]["hard"]/length_data, lor(data["length_by_difficulty"][result]["hard"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["hard"], length_data, alpha))
            print("---"*10)

        elif "explicit_no_reasoning" in result:
            print("Standard + Explicit + Feedback")
            print("Easy", data["length_by_difficulty"][result]["easy"]/length_data, lor(data["length_by_difficulty"][result]["easy"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["easy"], length_data, alpha))
            print("Medium", data["length_by_difficulty"][result]["medium"]/length_data, lor(data["length_by_difficulty"][result]["medium"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["medium"], length_data, alpha))
            print("Hard", data["length_by_difficulty"][result]["hard"]/length_data, lor(data["length_by_difficulty"][result]["hard"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["hard"], length_data, alpha))
            print("---"*10)

        elif "implicit_no_correction" == result:
            print("COT + Implicit + No Feedback")
            print("Easy", data["length_by_difficulty"][result]["easy"]/length_data, lor(data["length_by_difficulty"][result]["easy"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["easy"], length_data, alpha))
            print("Medium", data["length_by_difficulty"][result]["medium"]/length_data, lor(data["length_by_difficulty"][result]["medium"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["medium"], length_data, alpha))
            print("Hard", data["length_by_difficulty"][result]["hard"]/length_data, lor(data["length_by_difficulty"][result]["hard"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["hard"], length_data, alpha))
            print("---"*10)

        elif "explicit_no_correction" == result:
            print("COT + Explicit + No Feedback")
            print("Easy", data["length_by_difficulty"][result]["easy"]/length_data, lor(data["length_by_difficulty"][result]["easy"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["easy"], length_data, alpha))
            print("Medium", data["length_by_difficulty"][result]["medium"]/length_data, lor(data["length_by_difficulty"][result]["medium"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["medium"], length_data, alpha))
            print("Hard", data["length_by_difficulty"][result]["hard"]/length_data, lor(data["length_by_difficulty"][result]["hard"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["hard"], length_data, alpha))
            print("---"*10)

        elif "implicit" in result:
            print("COT + Implicit + Feedback")
            print("Easy", data["length_by_difficulty"][result]["easy"]/length_data, lor(data["length_by_difficulty"][result]["easy"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["easy"], length_data, alpha))
            print("Medium", data["length_by_difficulty"][result]["medium"]/length_data, lor(data["length_by_difficulty"][result]["medium"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["medium"], length_data, alpha))
            print("Hard", data["length_by_difficulty"][result]["hard"]/length_data, lor(data["length_by_difficulty"][result]["hard"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["hard"], length_data, alpha))
            print("---"*10)

        elif "explicit" in result:
            print("COT + Explicit + Feedback")
            print("Easy", data["length_by_difficulty"][result]["easy"]/length_data, lor(data["length_by_difficulty"][result]["easy"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["easy"], length_data, alpha))
            print("Medium", data["length_by_difficulty"][result]["medium"]/length_data, lor(data["length_by_difficulty"][result]["medium"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["medium"], length_data, alpha))
            print("Hard", data["length_by_difficulty"][result]["hard"]/length_data, lor(data["length_by_difficulty"][result]["hard"], length_data, alpha), 1 - lor(length_data - data["length_by_difficulty"][result]["hard"], length_data, alpha))
            print("---"*10)


if __name__ == "__main__":

    path = "results/for_analysis"

    for data_file in os.listdir(path):
        if data_file.endswith(".json"):
            with open(os.path.join(path, data_file), "r") as f:
                analyse(f)