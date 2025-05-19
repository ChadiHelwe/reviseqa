import os
import re
import shutil

from src.utils import read_file

pattern = re.compile(r'\(\s*p_\d+\([^)]+\)\s*<->\s*(?!-|~|¬)p_\d+\([^)]+\)\s*\)')



def biconditional_wrong_xor_check(formula):
    """
    Check if the formula contains an XOR operation.
    """
    # Check if the formula contains an XOR operation
    if pattern.search(formula):
        return True
    else:
        return False
    



def read_formula(file_path):
    """
    Read the formula from a file.
    """
    
    data  = read_file(file_path)

    cnt_incrementation = 1
    found_wrong = False
    for i in data["edits"]:
        forumula = i["prove9_input"]["formulas(assumptions)"]
        if cnt_incrementation == i["edit_number"]:
            for item in forumula:
                if biconditional_wrong_xor_check(item):
                    found_wrong = True
                    print(f"Found XOR operation in formula: {item}")
                    break

        
            goal_formulas = i["prove9_input"]["formulas(goals)"]
            for item in goal_formulas:
                if biconditional_wrong_xor_check(item):
                    found_wrong= True
                    print(f"Found XOR operation in formula: {item}")
                    break
        else:
            found_wrong = True
            break
        cnt_incrementation += 1

    return found_wrong




def main():
    """Main function for XOR check verification"""
    directory_nl_verified = "reviseqa_data/nl/verified"
    xor_verified = "reviseqa_data/nl/xor_verified"
    
    # Create output directory if it doesn't exist
    os.makedirs(xor_verified, exist_ok=True)
    
    cnt = 0
    all_cnt = 0
    for file in os.listdir(directory_nl_verified):
        file_path = os.path.join(directory_nl_verified, file)
        if "truncated" not in file:
            if os.path.isfile(file_path):
                print(f"Checking file: {file}")
                found_wrong_xor = read_formula(file_path)
                if found_wrong_xor:
                    print(f"Found XOR operation in file: {file}")
                    cnt += 1

                else:
                    print(f"No XOR operation found in file: {file}")
                    new_path = os.path.join(xor_verified, file)
                    shutil.copy(file_path, new_path)
            all_cnt += 1

    print(f"Total files with XOR operation: {cnt}")
    print(f"Total files checked: {all_cnt}")


if __name__ == "__main__":
    main()
    
