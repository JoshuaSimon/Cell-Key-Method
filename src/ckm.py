# ckm.py
# This python scripts implements the cell key method used 
# for a toy example. 
# --------------------------------------------------------
# Joshua Simon, 11.05.2022


import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 

# Values for the overlay matrix and vector are taken from 
# "Die Cell-Key-Methode – ein Geheimhaltungsverfahren" 
# by Jörg Höhne und Julia Höninger.
OVERLAY_MATRIX = np.matrix([
    [0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0.6875, 0.6875, 0.6875, 0.9375, 1, 1],
    [0, 0, 0.3533, 0.3533, 0.3533, 0.9440, 0.9970, 0.9990, 1],
    [0, 0.1620, 0.1620, 0.1620, 0.6620, 0.8560, 0.9970, 0.9990, 1],
    [0.0870, 0.0870, 0.0870, 0.1920, 0.6920, 0.8590, 0.9970, 0.9990, 1],
    [0, 0, 0.1450, 0.3270, 0.8270, 0.8590, 0.8930, 0.9490, 1],
    [0, 0.0400, 0.1500, 0.2850, 0.7850, 0.8600, 0.9200, 0.9600, 1],
    [0.0200, 0.0600, 0.1450, 0.2500, 0.7500, 0.8550, 0.9400, 0.9800, 1]
])

CHANGE_VECTOR = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def matrix_plot(matrix, vector):
    sns.heatmap(matrix, annot=True, xticklabels=vector, cbar=False)
    plt.xlabel("Überlagerungen")
    plt.ylabel("Originalwerte")
    plt.show()


def generate_data(n, seed):
    """ 
    Generates some random data from sample attributes. 
    Each row gets a random uniformly distributed record key
    between 0 and 1. 
    """
    np.random.seed(seed)
    universities = ["Bamberg", "Wuerzburg", "Muenchen", "Eichstaett"]
    sex = ["m", "w"]

    uni_data = np.random.choice(universities, size=n, replace=True, p=[0.15, 0.3, 0.5, 0.05])
    sex_data = np.random.choice(sex, size=n, replace=True, p=[0.5, 0.5])
    record_key_data = np.random.uniform(low=0.0, high=1.0, size=n)

    return pd.DataFrame(
        list(zip(uni_data, sex_data, record_key_data)),
        columns =['university', 'sex', 'record_key']
    )


def tabulate_data(data, rollout=False):
    """
    Generates the grouped frequency table with summed record keys.
    If the rollout option is true, all of the grouped sums are
    calculated as well.
    """
    grouped_data = data.groupby(["university", "sex"]).agg(["count", "sum"])
    grouped_data.columns = ["count", "record_key_sum"]
    grouped_data.reset_index(inplace=True)

    if rollout:
        rollout_data = data.loc[:, data.columns != "sex"].groupby(["university"]).agg(["count", "sum"])
        rollout_data.columns = ["count", "record_key_sum"]
        rollout_data.reset_index(inplace=True)
        rollout_data["sex"] = "i"
        rollout_data = rollout_data.iloc[:, [0,3,1,2]]

        sum_col = pd.DataFrame({
            "university": ["sum"],
            "sex": ["i"],
            "count": [grouped_data["count"].sum()],
            "record_key_sum": [grouped_data["record_key_sum"].sum()]
        })

        grouped_data = grouped_data.append([rollout_data, sum_col], ignore_index=True)
        grouped_data = grouped_data.sort_values(by=["university", "sex"])

    return grouped_data


def get_cell_key(value: float) -> float:
    """ 
    Returns the decimal part of a floating point number. 
    """
    return value - int(value)


def get_len_of_int(value: int) -> int:
    """
    Returns the length (= number of digits) of an positive integer.
    """
    return int(math.log10(value)) + 1


def get_overlay_matrix_value(matrix, vector, values, record_key_sums, seed, p0=1) -> list:
    """
    Returns the overlay value given by the overlay matrix and vector
    for a value-record_key_sum-pair.
    The overlay value is determined by the value ifself and the floating
    point digits of the record_key_sum value. The value is used as a 
    row-index to find the row in the overlay matrix. If the value and 
    therefore the row-index is out of range, the last row of the matrix
    is used. In the selected row, the index of the column, where the 
    record_key_sum is bigger than the column value is then used as in index
    for the overlay vector. The selected value of this vector is the
    overlay value which is to add to the original table value. The probability
    p0 determines the chance, that the overlay value is actually used.
    """
    np.random.seed(seed)
    overlay_col = []
    num_rows, _ = matrix.shape

    for value, record_key_sum in zip(values, record_key_sums):
        if value == 0:
            overlay_col.append(value)
            continue
        elif value < num_rows:
            cell_keys = matrix[value, :]
        else:
            cell_keys = matrix[num_rows - 1, :]

        for index, key in enumerate(cell_keys.tolist()[0]):
            if key > get_cell_key(record_key_sum):
                overlay_value = vector[index]
                break
        else:
            overlay_value = vector[-1]

        if p0 is not None:
            overlay_value = np.random.choice([overlay_value, 0], size=1, p=[1 - p0, p0])[0]
        overlay_col.append(overlay_value)

    return overlay_col


def apply_ckm(data, matrix, vector, value_col_names, record_key_names, seed, p) -> pd.DataFrame:
    """
    Applys the Cell Key Method to the named columns of a data set.
    Therefore the overlay value is calculated and added to the named
    columns.
    Returns a DataFrame with the overlayed data.
    """
    output_data = data.copy()
    for col_name, record_key_name, p0 in zip(value_col_names, record_key_names, p):
        output_data[col_name] = data[col_name] + get_overlay_matrix_value(matrix, vector, data[col_name], data[record_key_name], seed, p0)
    return output_data


if __name__ == "__main__":
    data = generate_data(1001, 42)
    table_data = tabulate_data(data)
    overlayed_data = apply_ckm(table_data, OVERLAY_MATRIX, CHANGE_VECTOR, ["count"], ["record_key_sum"], seed=42, p=[0])

    #print(data)
    print(table_data)
    print(overlayed_data)

    matrix_plot(OVERLAY_MATRIX, CHANGE_VECTOR)
    
    #print(get_cell_key(2.456))
    #print(get_len_of_int(1000))
    #print(get_overlay_matrix_value(OVERLAY_MATRIX, CHANGE_VECTOR, 251, 120.846))

    print("Done.")
