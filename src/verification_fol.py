import os
import shutil
from collections import Counter

from src.utils import read_file


def check_incrementation_edits(file_path):
    """
    Read the formula from a file.
    """

    data = read_file(file_path)

    cnt_incrementation = 1
    wrong_information = False
    for i in data["edits_made"]:
        if cnt_incrementation != i["Edit#"]:
            print(f"Wrong incrementation in file: {file_path}")
            wrong_information = True
            break
        else:
            cnt_incrementation += 1

    return wrong_information, cnt_incrementation


def exist_match_fol_strings(fol, list_fol):
    """
    Check if the fol string is in the list of fol strings.
    """
    exist = False
    for i in list_fol:
        i = i.replace("(", "").replace(")", "").replace(".", "")
        fol = fol.replace("(", "").replace(")", "").replace(".", "")
        if i == fol:
            exist = True
            break

    return exist


def check_error_fol_context_with_edits(file_path):
    """
    Read the formula from a file.
    """

    data = read_file(file_path)
    error = False
    for edit in data["edits_made"]:
        fol_context = edit["Edited Assumptions"].split("\n")
        removed_fol_facts = edit["Edits Made"]["removed_facts"]
        removed_fol_rules = edit["Edits Made"]["removed_rules"]
        added_fol_facts = edit["Edits Made"]["added_facts"]
        added_fol_rules = edit["Edits Made"]["added_rules"]

        # print(fol_context)
        # print(removed_fol_facts)
        # print(removed_fol_rules)
        # print(added_fol_facts)
        # print(added_fol_rules)
        # print("=====================================")

        for i in removed_fol_facts:
            if exist_match_fol_strings(i, fol_context):
                print("----------------------------------")
                print(fol_context)
                print(removed_fol_facts)
                print(removed_fol_rules)
                print(added_fol_facts)
                print(added_fol_rules)
                print("--------------------------------")
                print("Found:", i)
                error = True

        for i in removed_fol_rules:
            if exist_match_fol_strings(i, fol_context):
                print("----------------------------------")
                print(fol_context)
                print(removed_fol_facts)
                print(removed_fol_rules)
                print(added_fol_facts)
                print(added_fol_rules)
                print("--------------------------------")
                print("Found:", i)
                error = True

        for i in added_fol_facts:
            if not exist_match_fol_strings(i, fol_context):
                print("----------------------------------")
                print(fol_context)
                print(removed_fol_facts)
                print(removed_fol_rules)
                print(added_fol_facts)
                print(added_fol_rules)
                print("--------------------------------")
                print("Not Found:", i)
                error = True

        for i in added_fol_rules:
            if not exist_match_fol_strings(i, fol_context):
                print("----------------------------------")
                print(fol_context)
                print(removed_fol_facts)
                print(removed_fol_rules)
                print(added_fol_facts)
                print(added_fol_rules)
                print("--------------------------------")
                print("Not Found:", i)
                error = True

    return error


def main():
    """Main function for FOL verification"""
    list_incrementations = []
    cnt = 0
    all_cnt = 0
    path = "reviseqa_data/fol"
    
    # Create output directory if it doesn't exist
    os.makedirs("reviseqa_data/verification_1_fol", exist_ok=True)
    
    for ex in os.listdir(path):
        example_path = os.path.join(path, ex)
        error_consistency = check_error_fol_context_with_edits(example_path)

        wrong_incrementation, cnt_incrementation = check_incrementation_edits(
            example_path
        )
        if not wrong_incrementation and not error_consistency:
            list_incrementations.append(cnt_incrementation - 1)
            shutil.copy(example_path, f"reviseqa_data/verification_1_fol/{ex}")
            cnt += 1
        all_cnt += 1

    print(Counter(list_incrementations))
    print("Correct files", cnt)
    print("All files", all_cnt)


if __name__ == "__main__":
    main()
