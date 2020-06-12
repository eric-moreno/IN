from keras.layers import BatchNormalization, Conv1D, SpatialDropout1D, Dropout, Concatenate, Dense, Input, GRU, Flatten
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

def FC(data, num_hidden, act='relu', p=None, name='', trainable=True):
    kernel_initializer_fc = 'lecun_uniform'
    if act=='leakyrelu':
        fc = Dense(num_hidden, activation='linear',
                   name='%s_%s' % (name,act),
                   kernel_initializer=kernel_initializer_fc,
                   trainable=trainable)(data) # Add any layer, with the default of a linear squashing function
        fc = LeakyReLU(alpha=.001)(fc)   # add an advanced activation                                                                                                     
    else:
        fc = Dense(num_hidden, activation=act,
                   name='%s_%s' % (name,act),
                   kernel_initializer=kernel_initializer_fc,
                   trainable=trainable)(data)
    if not p:
        return fc
    else:
        dropout = Dropout(rate=p, name='%s_dropout' % name)(fc)
        return dropout

def model_DeepDoubleXReference(inputs, num_classes, datasets = ['cpf','sv'], hlf_input = None, scale_hidden = 1,  **kwargs):
    """
    reference 1x1 convolutional model for 'deepDoubleX'
    with recurrent layers and batch normalisation
    """

    kernel_initializer = 'he_normal'
    kernel_initializer_fc = 'lecun_uniform'

    normalizedInputs = []
    for i in range(len(inputs)):
        normedLayer = BatchNormalization(momentum=0.3,axis=-1, name = '%s_input_batchnorm'%datasets[i])(inputs[i])
        normalizedInputs.append(normedLayer)

    flattenLayers = []
    if hlf_input is not None:
        flattenLayers.append(Flatten()(normalizedInputs[hlf_input]))
        
    for i in range(len(inputs)):
        if i==hlf_input: continue
        ds = datasets[i]
        normalizedInput = normalizedInputs[i]
        x = Conv1D(filters=32*scale_hidden, kernel_size=(1,), strides=(1,), padding='same', 
                                kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv1'%ds, 
                                activation = 'relu', data_format='channels_last')(normalizedInput)
        x = SpatialDropout1D(rate=0.1)(x)
        x = Conv1D(filters=32*scale_hidden, kernel_size=(1,), strides=(1,), padding='same', 
                             kernel_initializer=kernel_initializer, use_bias=False, name='%s_conv2'%ds, 
                             activation = 'relu', data_format='channels_last')(x)
        x = SpatialDropout1D(rate=0.1)(x)
        x = GRU(50*scale_hidden,go_backwards=True,implementation=2,name='%s_gru'%ds)(x)
        x = Dropout(rate=0.1)(x)
        flattenLayers.append(x)

    if len(datasets)>1: 
        concat = Concatenate()(flattenLayers)
    else: 
        concat = flattenLayers[0]

    fc = FC(concat, 100*scale_hidden, p=0.1, name='fc1')
    output = Dense(num_classes, activation='softmax', name='ID_pred', kernel_initializer=kernel_initializer_fc)(fc)
                            
    model = Model(inputs=inputs, outputs=[output])

    return model

