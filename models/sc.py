import tensorflow as tf
from tensorflow import keras, nn
from tensorflow.keras import losses

from models.custom_losses import bn_loss


class StyleContent(keras.Model):
    def call(self, input):
        global global_bsz
        target, gen = input
        orig_gen = gen

        # Preprocess
        target = self.preprocess_input((target + 1) * 127.5)
        gen = self.preprocess_input((gen + 1) * 127.5)

        # Features
        target_feats = self.cnn(target)
        gen_feats = self.cnn(gen)

        # Style loss
        style_layer_losses = [bn_loss(t, g) for t, g in
                              zip(target_feats['style'], gen_feats['style'])]
        style_loss = sum(style_layer_losses) / len(style_layer_losses)
        style_loss = nn.compute_average_loss(style_layer_losses,
                                             global_batch_size=global_bsz)

        # Content loss
        content_loss = losses.mse(target_feats['content'], gen_feats['content'])
        content_loss = tf.reduce_mean(content_loss, axis=[1, 2])
        content_loss = nn.compute_average_loss(content_loss,
                                               global_batch_size=global_bsz)

        self.add_loss(content_loss + 0.1 * style_loss)
        self.add_metric(style_loss, 'style-loss')
        self.add_metric(content_loss, 'content-loss')
        return orig_gen


class SC_VGG19(StyleContent):
    def __init__(self, args):
        super().__init__()
        self.preprocess_input = keras.applications.vgg19.preprocess_input
        cnn = keras.applications.VGG19(include_top=False)
        cnn.trainable = False

        if args.style_layer > 0:
            style_outs = [cnn.get_layer(f'block{i}_conv1').output
                          for i in range(1, args.style_layer + 1)]
        else:
            style_outs = [cnn.input]

        if args.content_layer > 0:
            content_out = cnn.get_layer(f'block{args.content_layer}_conv2').output
        else:
            content_out = cnn.input

        self.cnn = keras.Model(cnn.input, outputs={'style': style_outs,
                                                   'content': content_out},
                               name='style-content')


class SC_VGG16(StyleContent):
    def __init__(self, args):
        super().__init__()
        self.preprocess_input = keras.applications.vgg16.preprocess_input
        cnn = keras.applications.VGG16(include_top=False)
        cnn.trainable = False

        style_outs = [cnn.get_layer(f'block{i}_conv1').output
                      for i in range(1, args.style_layer + 1)]

        content_out = cnn.get_layer(f'block{args.content_layer}_conv2').output

        self.cnn = keras.Model(cnn.input, outputs={'style': style_outs,
                                                   'content': content_out},
                               name='style-content')
