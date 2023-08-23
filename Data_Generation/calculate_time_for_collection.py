def calculate_time_for_overall_data_collection(number_of_samples, time_per_sample):
    """
    Calculate the time needed for the overall data collection
    :param number_of_samples:
    :param time_per_sample: in seconds
    :return:
    """

    total_time = number_of_samples * time_per_sample
    print("Total time for data collection: ", total_time)
    print("Total time for data collection in minutes: ", total_time / 60)
    print("Total time for data collection in hours: ", total_time / 3600)
    print("Total time for data collection in days: ", total_time / 86400)
    print("Total time for data collection in weeks: ", total_time / 604800)
    print("Total time for data collection in months: ", total_time / 2628000)
    print("Total time for data collection in years: ", total_time / 31536000)


if __name__ == '__main__':
    calculate_time_for_overall_data_collection(number_of_samples=3000, time_per_sample=20)
