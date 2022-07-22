from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(complete_x_list,
                                                    complete_y_list,
                                                    test_size=0.2,
                                                    shuffle=True)

X_train = np.array(X_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)

model = dl_models.get_model_lstm(dense_input, dense_input_dim,
                                 dense_input_activation,
                                 dense_output_activation,
                                 n_lstm_units, loss_type,
                                 optimizer_type, 1, dropout_value)