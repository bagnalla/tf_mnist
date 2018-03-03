from dataset_params import emnist_load_data
import pickle

train, validation, test = emnist_load_data()

with open("emnist/train.pkl", "wb") as f:
    pickle.dump(train, f)

with open("emnist/validation.pkl", "wb") as f:
    pickle.dump(validation, f)

with open("emnist/test.pkl", "wb") as f:
    pickle.dump(test, f)
