import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')


train['Embarked'] = pd.Categorical(train['Embarked'])
train['Embarked'] = train['Embarked'].cat.codes

train['Age'] = train['Age'].apply(lambda x: 27.0 if np.isnan(x)==True else x)
train['Age_cat'] = train['Age'].apply(lambda x: '0-6' if x<=6 else ('6-12' if x<=12 else ('12-18') if x<=18 else ('18-24') if x<=24 else ('24-30' if x<=30 else ('30-40' if x<=40 else ('40-55' if x<=55 else '55+')))))
train['Age_cat'] = pd.Categorical(train['Age_cat'])
train['Age_cat'] = train['Age_cat'].cat.codes

train['Family'] = train['SibSp'] + train['Parch']
train['Family'] = train['Family'].apply(lambda x: 1 if x>0 else 0)

train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.')
train['Title'] = train['Title'].apply(lambda x: 'Miss' if (x == 'Ms')|(x=='Miss')|(x=='Mlle') else ('Mrs' if (x == 'Mrs')|(x=='Mme') else ('Mr' if x=='Mr' else ('Master' if x=='Master' else 'Other'))))
train['Title'] = pd.Categorical(train['Title'])
train['Title'] = train['Title'].cat.codes

train['Sex'] = train['Sex'].apply(lambda x: 0 if x=='male' else 1)


X = train[['Sex', 'Pclass', 'Fare', 'Embarked', 'Title', 'Family']]
y = train['Survived']


X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y, train_size=0.8)

input_shape = [X_train.shape[1]]



model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Dense(16, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid'),
])


model.summary()




model.compile(optimizer='adam', loss='binary_crossentropy',
    metrics=['binary_accuracy'],)


# 提早停止
early_stopping = keras.callbacks.EarlyStopping(
    patience=8,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=1000,
    batch_size=512,
    callbacks=[early_stopping],
)



history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")

# 绘制训练损失和验证损失
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Losses')
plt.legend()

# 绘制训练准确率和验证准确率
plt.subplot(1, 2, 2)
plt.plot(history.history['binary_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
plt.title('Accuracies')
plt.legend()

plt.show()


test['Age'].median()
test['Sex'] = test['Sex'].apply(lambda x: 0 if x=='male' else 1)
test['Age'] = test['Age'].apply(lambda x: 27.0 if np.isnan(x)==True else x)
test['Age_cat'] = test['Age'].apply(lambda x: '0-6' if x<=6 else ('6-12' if x<=12 else ('12-18') if x<=18 else ('18-24') if x<=24 else ('24-30' if x<=30 else ('30-40' if x<=40 else ('40-55' if x<=55 else '55+')))))
test['Age_cat'] = pd.Categorical(test['Age_cat'])
test['Age_cat'] = test['Age_cat'].cat.codes
test['Embarked'] = pd.Categorical(test['Embarked'])
test['Embarked'] = test['Embarked'].cat.codes
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.')
test['Title'] = test['Title'].apply(lambda x: 'Miss' if (x == 'Ms')|(x=='Miss')|(x=='Mlle') else ('Mrs' if (x == 'Mrs')|(x=='Mme') else ('Mr' if x=='Mr' else ('Master' if x=='Master' else 'Other'))))
test['Title'] = pd.Categorical(test['Title'])
test['Title'] = test['Title'].cat.codes
test['Family'] = test['SibSp'] + test['Parch']
test['Family'] = test['Family'].apply(lambda x: 1 if x>0 else 0)
test_X = test[['Sex', 'Pclass', 'Fare', 'Embarked', 'Title', 'Family']]




pd.DataFrame(test_X).fillna(0, inplace=True)
median = test_X.median()
test_X.loc[:, 'Fare'] = test_X['Fare'].fillna(median)



df = pd.DataFrame()
test_X = StandardScaler().fit_transform(test_X)
df['PassengerId'] = test['PassengerId']
df['Survived'] = model.predict(test_X).round().astype(int)



df.to_csv('submission.csv',index=False)
