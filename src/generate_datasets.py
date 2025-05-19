from src.generate_reviseqa import make_dataset, parallel_make_dataset
from src.generate_reviseqa_nl import parallel_make_dataset_nl

if __name__ == "__main__":
    # parallel_make_dataset("provergen_data/translated_data/hard-500-0_500.json")

    parallel_make_dataset_nl("reviseqa_data/verification_1_fol")
# print(parallel_make_dataset("reviseqa_data/fol"))
