# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Input
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib
#
#
# class DicePredictor:
#     def __init__(self, model_type='lstm'):
#         self.model_type = model_type
#         self.model = None
#
#     def build_hybrid_model(self, input_shape):
#         input_layer = Input(shape=input_shape)
#
#         # CNN特征提取
#         x = Conv2D(32, (3, 3), activation='relu')(input_layer)
#         x = MaxPooling2D()(x)
#         x = Conv2D(64, (3, 3), activation='relu')(x)
#         x = MaxPooling2D()(x)
#
#         # LSTM时序处理
#         x = TimeDistributed(Flatten())(x)
#         x = LSTM(64)(x)
#
#         # 输出层
#         output = Dense(6, activation='softmax')(x)
#
#         model = Model(inputs=input_layer, outputs=output)
#         model.compile(
#             loss='sparse_categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy']
#         )
#
#         return model
#
#     def build_lstm_model(self, input_shape):
#         """构建LSTM模型"""
#         model = Sequential()
#         model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
#         model.add(Dropout(0.2))
#         model.add(LSTM(32))
#         model.add(Dropout(0.2))
#         model.add(Dense(16, activation='relu'))
#         model.add(Dense(6, activation='softmax'))  # 6个可能的骰子结果
#
#         model.compile(
#             loss='sparse_categorical_crossentropy',
#             optimizer='adam',
#             metrics=['accuracy']
#         )
#
#         return model
#
#     def train(self, X, y, test_size=0.2, epochs=50, batch_size=32):
#         """训练模型"""
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=test_size, random_state=42
#         )
#
#         if self.model_type == 'lstm':
#             # 对于LSTM，需要调整输入形状
#             if len(X.shape) < 3:
#                 # 如果不是序列数据，转换为序列形状 (samples, timesteps, features)
#                 X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
#                 X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
#
#             self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
#
#             # 训练模型
#             history = self.model.fit(
#                 X_train, y_train - 1,  # 减1是因为骰子点数从1开始，但索引从0开始
#                 validation_data=(X_test, y_test - 1),
#                 epochs=epochs,
#                 batch_size=batch_size
#             )
#
#             # 评估模型
#             loss, accuracy = self.model.evaluate(X_test, y_test - 1)
#             print(f"测试集准确率: {accuracy:.4f}")
#
#             return history, accuracy
#
#         elif self.model_type == 'rf':
#             # 随机森林模型
#             self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#             self.model.fit(X_train, y_train)
#
#             # 评估模型
#             accuracy = self.model.score(X_test, y_test)
#             print(f"测试集准确率: {accuracy:.4f}")
#
#             return None, accuracy
#
#         elif self.model_type == 'hybrid':
#             # 对于CNN+LSTM，需要调整输入形状
#             if len(X.shape) < 4:
#                 # 如果不是序列数据，转换为序列形状 (samples, timesteps, height, width, channels)
#                 X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2], X_train.shape[3])
#                 X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2], X_test.shape[3])
#
#             self.model = self.build_hybrid_model(
#                 (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4]))
#
#             # 训练模型
#             history = self.model.fit(
#                 X_train, y_train - 1,  # 减1是因为骰子点数从1开始，但索引从0开始
#                 validation_data=(X_test, y_test - 1),
#                 epochs=epochs,
#                 batch_size=batch_size
#             )
#
#             # 评估模型
#             loss, accuracy = self.model.evaluate(X_test, y_test - 1)
#             print(f"测试集准确率: {accuracy:.4f}")
#
#             return history, accuracy
#
#     def predict(self, X):
#         """预测骰子点数"""
#         if self.model is None:
#             raise ValueError("模型尚未训练")
#
#         if self.model_type == 'lstm' and len(X.shape) < 3:
#             # 调整输入形状
#             X = X.reshape(X.shape[0], 1, X.shape[1])
#
#             # 预测并转换回1-6
#             predictions = self.model.predict(X)
#             return np.argmax(predictions, axis=1) + 1
#         elif self.model_type == 'hybrid' and len(X.shape) < 4:
#             # 调整输入形状
#             X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])
#
#             # 预测并转换回1-6
#             predictions = self.model.predict(X)
#             return np.argmax(predictions, axis=1) + 1
#         else:
#             return self.model.predict(X)
#
#     def save_model(self, path):
#         """保存模型"""
#         if self.model is None:
#             raise ValueError("模型尚未训练")
#
#         if self.model_type == 'lstm':
#             self.model.save(path)
#         elif self.model_type == 'hybrid':
#             self.model.save(path)
#         else:
#             joblib.dump(self.model, path)
#
#     def load_model(self, path):
#         """加载模型"""
#         if self.model_type == 'lstm':
#             self.model = tf.keras.models.load_model(path)
#         elif self.model_type == 'hybrid':
#             self.model = tf.keras.models.load_model(path)
#         else:
#             self.model = joblib.load(path)
