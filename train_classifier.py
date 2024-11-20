import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))
for i, item in enumerate(data_dict['data']):
    print(f"Item {i} has length: {len(item)}")
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])


max_length = 84  # The maximum length in your data

# Pad each item to match the max_length
data = [np.pad(item, (0, max_length - len(item))) if len(item) < max_length else item for item in data_dict['data']]

# Convert to numpy array
data = np.asarray(data)

print(f"Data shape: {data.shape}")
labels = np.asarray(data_dict['labels'])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
