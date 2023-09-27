import os

def create_results_path(case_name):
    case_name = f'{case_name}'
    results_f_n = f'results'
    if case_name == '':
        total_path = f'{results_f_n}'
        try:
                os.mkdir(total_path)
        except FileExistsError:
              pass  
    else:
        total_path = f'{case_name}\{results_f_n}'
        try:
                os.makedirs(total_path)
        except FileExistsError:
              pass  
    # now the path to results is .\results
    return total_path

def create_data_path(case_name):
    case_name = f'{case_name}'
    data_f_n = f'data'
    if case_name == '':
        total_path = f'{data_f_n}'
        try:
                os.mkdir(total_path)
        except FileExistsError:
              pass  
    else:
        total_path = f'{case_name}\{data_f_n}'
        try:
                os.makedirs(total_path)
        except FileExistsError:
              pass  
    # now the path to data is .\data
    return total_path

# os.mkdir('tempDir')