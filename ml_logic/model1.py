from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense,BatchNormalization, \
    concatenate, Flatten, Activation, Input,  Conv3DTranspose
from tensorflow.keras.models import  Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def glioma_label_unet_model():
    #define a 3D Unet modell
    in_layer = Input((190, 190, 128, 1))
    bn = BatchNormalization()(in_layer)
    cn1 = Conv3D(8,
                kernel_size = (1, 5, 5),
                padding = 'same',
                activation = 'relu')(bn)
    bn2 = Activation('relu')(BatchNormalization()(cn1))

    dn1 = MaxPooling3D((2, 2, 2))(bn2)
    cn3 = Conv3D(16,
                kernel_size = (3, 3, 3),
                padding = 'same',
                activation = 'relu')(dn1)
    bn3 = Activation('relu')(BatchNormalization()(cn3))

    dn2 = MaxPooling3D((1, 2, 2))(bn3)
    cn4 = Conv3D(32,
                kernel_size = (3, 3, 3),
                padding = 'same',
                activation = 'relu')(dn2)
    bn4 = Activation('relu')(BatchNormalization()(cn4))

    up1 = Conv3DTranspose(16,
                        kernel_size = (3, 3, 3),
                        strides = (1, 2, 2),
                        padding = 'same')(bn4)

    cat1 = concatenate([up1, bn3], axis=2)

    up2 = Conv3DTranspose(8,
                        kernel_size = (3, 3, 3),
                        strides = (2, 2, 2),
                        padding = 'same')(cat1)

    pre_out = concatenate([up2, bn2], axis=2)

    #pre_out
    pre_out = Conv3D(1,
                kernel_size = (1, 1, 1),
                padding = 'same',
                activation = 'relu')(pre_out)


    pre_out = Flatten()(pre_out)

    pre_out = Dense(32, activation = 'relu')(pre_out)
    out = Dense(1, activation='sigmoid')(pre_out)

    sim_model = Model(inputs = [in_layer], outputs = [out])
    return sim_model

def fit_complie_model(model,X_train,y_train,
                           learning_rate= 0.001, metrics = ['accuracy']
                           ,patience=5):
    optim=Adam(learning_rate)
    model.compile(loss = 'binary_crossentropy',
                  optimizer = optim,
                  metrics = metrics)
    es = EarlyStopping(patience=patience, restore_best_weights = True)
    history = model.fit(X_train, y_train,
                        epochs = 30,
                        batch_size = 4,
                        callbacks = [es],
                        validation_split=0.2,
                        shuffle =True,
                        verbose = 1)
    return history
