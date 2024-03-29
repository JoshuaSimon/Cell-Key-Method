import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 

# Werte der  Überlagerungsmatrix und des Änderungsvektors stammen
# aus "Die Cell-Key-Methode – ein Geheimhaltungsverfahren" 
# von Jörg Höhne und Julia Höninger.
OVERLAY_MATRIX = np.matrix([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
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
    """
    Heatmap der Überlagerungsmatrix.
    """
    sns.heatmap(matrix, annot=True, xticklabels=vector, cbar=False)
    plt.xlabel("Überlagerungen")
    plt.ylabel("Originalwerte")
    plt.show()


def generate_data(n, seed) -> pd.DataFrame:
    """ 
    Erstellt zufällige Testdaten von einigen Merkmalsaus-
    pärgungen. Jeder Eintrag wird mit einem gleichverteil-
    ten Recordkey zwischen 0 und 1 versehen.
    """
    np.random.seed(seed)
    universities = [
        "Bamberg", "Wuerzburg",
        "Muenchen", "Eichstaett"]
    sex = ["m", "w"]

    uni_data = np.random.choice(universities, size=n, replace=True, p=[0.15, 0.3, 0.5, 0.05])
    sex_data = np.random.choice(sex, size=n, replace=True, p=[0.5, 0.5])
    record_key_data = np.random.uniform(low=0.0, high=1.0, size=n)

    return pd.DataFrame(
        list(zip(uni_data, sex_data, record_key_data)),
        columns =["university", "sex", "record_key"]
    )


def tabulate_data(data, rollout=False):
    """
    Gibt die gruppierten Daten als Häufigkeitstablle mit den 
    Summierten Recordkeys zurück. 
    Wenn die `rollout` Option gesetzt ist, werden ebenfalls
    alle Gruppensummen mit ausgegeben. 
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

        grouped_data = grouped_data.append(
            [rollout_data, sum_col], ignore_index=True)
        grouped_data = grouped_data.sort_values(
            by=["university", "sex"])

    return grouped_data


def get_cell_key(value: float) -> float:
    """ 
    Gibt den Dezimalanteil ein Gleitkommazahl zurück.
    """
    return value - int(value)


def get_overlay_matrix_value(matrix, vector, values, record_key_sums, seed, p0=1) -> list:
    """
    Gibt eine Liste aus Überlagerungswerten passend zu den Paaren 
    aus Originalwerten und Recordkey-Summen zurück. Aus den 
    Recordkey-Summen werden die Cellkeys bestimmt. 
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
    Wendet die Cell Key Methode auf die angegebenen Spalten des 
    Dataframes an. Hierfür wird der Überlagerungswert berechnet
    und zu den Spalten addiert. 
    Gibt einen Dateframe mit den überlagerten Werten zurück.
    """
    output_data = data.copy()
    for col_name, record_key_name, p0 in zip(value_col_names, record_key_names, p):
        output_data[col_name] = data[col_name] + get_overlay_matrix_value(matrix, vector, data[col_name], data[record_key_name], seed, p0)
    return output_data


if __name__ == "__main__":
    data = generate_data(1001, 42)
    table_data = tabulate_data(data)
    overlayed_data = apply_ckm(
        table_data,
        OVERLAY_MATRIX,
        CHANGE_VECTOR,
        ["count"], ["record_key_sum"],
        seed=42, p=[0])

    # Erstellen der Hearmap.
    matrix_plot(OVERLAY_MATRIX, CHANGE_VECTOR)

    # Darstellen der Testdaten, der tabellierten Testdaten und der 
    # geheimgehaltenen Testdaten.
    print(data)
    print(table_data)
    print(overlayed_data)
