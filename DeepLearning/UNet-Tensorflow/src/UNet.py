import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Dense, Layer, Activation, MaxPool2D
from tensorflow.keras.models import Model


# def ConvBnRelu2D(in_channels, out_channels, kernel_size=(3, 3), padding=1):
#     model = keras.Sequential()
#     model.add(Conv2D(out_channels, kernel_size, padding='same', use_bias=False))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     return model
#
#
# class StackEncoder(tf.Module):
#     def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
#         super(StackEncoder, self).__init__()
#         self.encode = keras.Sequential()
#         self.encode.add(ConvBnRelu2D(x_channels, y_channels, kernel_size=kernel_size))
#         self.encode.add(ConvBnRelu2D(y_channels, y_channels, kernel_size=kernel_size))
#
#     def __call__(self, inputs):
#         x = self.encode(inputs)
#         x_small = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
#         return x, x_small
#
#
# class StackDecoder(tf.Module):
#     def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
#         super(StackDecoder, self).__init__()
#         padding = (kernel_size - 1) // 2
#         self.up = Conv2DTranspose(x_channels, kernel_size=(2, 2), strides=(2, 2))
#
#         self.decode = keras.Sequential()
#         self.decode.add(ConvBnRelu2D(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding))
#         self.decode.add(ConvBnRelu2D(y_channels, y_channels, kernel_size=kernel_size, padding=padding))
#         self.decode.add(ConvBnRelu2D(y_channels, y_channels, kernel_size=kernel_size, padding=padding))
#
#     def __call__(self, x, down_tensor):
#         # print(down_tensor.shape)
#         _, height, width, channels = down_tensor.shape
#         x = self.up(x)
#         x = tf.concat([x, down_tensor], axis=3)
#         x = self.decode(x)
#         return x
#
#
# class UNet(Model):
#
#     def __init__(self, c_in=1, num_classes=3):
#         super(UNet, self).__init__()
#
#         self.down1 = StackEncoder(c_in, 24, kernel_size=3)  # 128
#         self.down2 = StackEncoder(24, 64, kernel_size=3)  # 64
#         self.down3 = StackEncoder(64, 128, kernel_size=3)  # 32
#         self.down4 = StackEncoder(128, 256, kernel_size=3)  # 16
#         self.down5 = StackEncoder(256, 512, kernel_size=3)  # 8
#
#         self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 16
#         self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 32
#         self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 64
#         self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 128
#         self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 256
#
#         self.classify = Conv2D(num_classes, kernel_size=1, use_bias=True)
#
#         self.center = keras.Sequential()
#         self.center.add(ConvBnRelu2D(512, 512, kernel_size=3, padding=1))
#
#     def call(self, x):
#         down1, out = self.down1(x)
#         down2, out = self.down2(out)
#         down3, out = self.down3(out)
#         down4, out = self.down4(out)
#         down5, out = self.down5(out)
#
#         out = self.center(out)
#
#         out = self.up5(out, down5)
#         out = self.up4(out, down4)
#         out = self.up3(out, down3)
#         out = self.up2(out, down2)
#         out = self.up1(out, down1)
#
#         out = self.classify(out)
#
#         return out


def ConvBatchNormRelu2D(x, c_out, kernel_size=(3, 3)):
    x = Conv2D(c_out, kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def StackEncoder(x, x_channel, y_channel, kernel_size=(3, 3)):
    x = ConvBatchNormRelu2D(x, y_channel, kernel_size)
    x = ConvBatchNormRelu2D(x, y_channel, kernel_size)
    x_small = MaxPool2D(pool_size = (2, 2), strides = (2, 2))(x)
    return x, x_small

def StackDecoder(x, down_tensor, x_big_channel, x_channel, y_channel, kernel_size=(3, 3)):
    x = Conv2DTranspose(x_channel, kernel_size=(2, 2), strides=(2, 2))(x)
    x = tf.concat([x, down_tensor], axis=3)

    x = ConvBatchNormRelu2D(x, y_channel, kernel_size=kernel_size)
    x = ConvBatchNormRelu2D(x, y_channel, kernel_size=kernel_size)
    x = ConvBatchNormRelu2D(x, y_channel, kernel_size=kernel_size)
    return x

def unet_functional(x, c_in = 3, num_classes=3):
    down1, out = StackEncoder(x, c_in, 24)
    down2, out = StackEncoder(out, 24, 64)
    down3, out = StackEncoder(out, 64, 128)
    down4, out = StackEncoder(out, 128, 256)
    down5, out = StackEncoder(out, 256, 512)

    out = ConvBatchNormRelu2D(out, 512)

    out = StackDecoder(out, down5, 512, 512, 256)
    out = StackDecoder(out, down4, 256, 256, 128)
    out = StackDecoder(out, down3, 128, 128, 64)
    out = StackDecoder(out, down2, 64, 64, 24)
    out = StackDecoder(out, down1, 24, 24, 24)

    out = Conv2D(num_classes, kernel_size=(1, 1), use_bias=True)(out)


    return out
