import numpy as np
import pandas as pd


# import main


def read_interpretation_label(machine, machine_2, row):
    file = open("data/SMD/interpretation_label/machine-" + str(machine) + "-" + str(machine_2) + ".txt", "r")
    lines = file.readlines()
    intepretation_labels = {}

    for line in lines:
        # Remove \n and split at :
        range_part, key_part = line.strip().split(":")

        # Split the keys at ,
        keys = key_part.split(",")

        for key in keys:
            key = int(key)
            if key not in intepretation_labels:
                intepretation_labels[key] = []
            intepretation_labels[key].append(range_part)

    interpretation_labels = intepretation_labels[row]

    anomaly_ranges = []
    for range in interpretation_labels:
        tmp = range.split("-")
        value1 = int(tmp[0])
        value2 = int(tmp[1])
        values = (value1, value2)
        anomaly_ranges.append(values)

    return anomaly_ranges


def read_train_machine(machine, machine_2, row):
    file = open("data/SMD/train/machine-" + str(machine) + "-" + str(machine_2) + ".txt", "r")
    lines = file.readlines()
    machine_data = []

    for line in lines:
        tmp = line.strip().split(",")
        machine_data.append(tmp[row - 1])

    return machine_data


def split_data(anomaly_ranges, machine_data):
    machine_data_copy = machine_data.copy()
    anomaly_ranges_copy = anomaly_ranges.copy()
    anomalies = []

    # Copy the anomalies from machine_data to anomalies list
    for range in anomaly_ranges_copy:
        start, end = range
        anomalies.append(machine_data[start:end + 1])

    # Reverse list of ranges so those ranges can be deleted in backwards order
    anomaly_ranges_copy.reverse()

    # Delete the anomalies from machine_data
    for range in anomaly_ranges_copy:
        start, end = range
        del machine_data_copy[start:end + 1]

    anomalies = [item for sublist in anomalies for item in sublist]  # Instead of Lists inside a List we need one List

    return machine_data_copy, anomalies


def create_normal_with_gaps_series_list(normals, number_normal, RANDOM_SEED,
                                        length=140):  # Creates a List of Timeseries which include random chosen Points of normal Data
    np.random.seed(RANDOM_SEED)

    normal_series_list = []

    while len(normal_series_list) < number_normal:
        rand_index = np.random.randint(0, len(normals) - length)
        tmp = []

        for i in range(length):
            tmp.append(normals[rand_index])
            rand_index += 1

        normal_series_list.append(tmp)

    return normal_series_list


def create_normal_series_list(machine_data, anomaly_ranges, number_normal, RANDOM_SEED, length=140):
    np.random.seed(RANDOM_SEED)

    normal_series_list = []

    while len(normal_series_list) != number_normal:
        anomalies_found = False
        rand_index = np.random.randint(0, len(machine_data) - length)
        tmp = []

        for x in anomaly_ranges:
            value1, value2 = x

            for i in range(length):
                if value1 <= rand_index + i <= value2:
                    anomalies_found = True
                    break

        if not anomalies_found:
            for i in range(length):
                tmp.append(machine_data[rand_index])
                rand_index += 1
            normal_series_list.append(tmp)

    return normal_series_list


def create_anomaly_series_list(machine_data, anomaly_ranges, number_anomalies,
                               RANDOM_SEED, length=140):
    np.random.seed(RANDOM_SEED)

    number_anomaly_points = int(0.4 * length)

    anomaly_series_list = []

    while len(anomaly_series_list) != number_anomalies:
        anomalies_found = 0
        rand_index = np.random.randint(0, len(machine_data) - length)
        tmp = []

        for tuple in anomaly_ranges:
            value1, value2 = tuple

            for i in range(length):
                if value1 <= rand_index + i <= value2:
                    anomalies_found += 1

        if (number_anomaly_points <= anomalies_found):
            for i in range(length):
                tmp.append(machine_data[rand_index])
                rand_index += 1
            anomaly_series_list.append(tmp)

    return anomaly_series_list


def create_panda_df(normal_series_list, anomaly_series_list):
    normal_df = pd.DataFrame(normal_series_list)
    normal_df["target"] = "1"

    anomaly_df = pd.DataFrame(anomaly_series_list)
    anomaly_df["target"] = "2"

    combined_df = pd.concat([normal_df, anomaly_df], ignore_index=True)

    return combined_df


# only used for debugging
def check_length(series_list, length):
    for i in series_list:
        if len(i) != length:
            print(len(i))
            return False
    return True


def main(config, RANDOM_SEED):
    # Use given yaml config
    machine_1 = config["data"]["smd_machine_1"]
    machine_2 = config["data"]["smd_machine_2"]
    col = config["data"]["smd_col"]
    normal_number = config["data"]["smd_normal_number"]
    anomaly_number = config["data"]["smd_anomaly_number"]
    length = config["data"]["smd_length"]

    # Read the Interpretation labels and save them
    interpretation_labels = read_interpretation_label(machine_1, machine_2, col)

    # Read machine data and save it
    machine_data = read_train_machine(machine_1, machine_2, col)

    # Create list of normal timeseries data
    normal_series_list = create_normal_series_list(machine_data, interpretation_labels, normal_number, RANDOM_SEED,
                                                   length=length)

    # Create list of timeseries data with anomalies
    anomaly_series_list = create_anomaly_series_list(machine_data, interpretation_labels, anomaly_number,
                                                     RANDOM_SEED, length=length)
    # Create the combined panda dataframe of normal and abnormal time series lists
    combined_df = create_panda_df(normal_series_list, anomaly_series_list)

    return combined_df


if __name__ == "__main__":
    main("", 0)
