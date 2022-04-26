class UnetParam(object):

    def __init__(self, nucleus_unet_model, actin_unet_model, unet_model_scale, unet_model_thrh, unet_img_size):
        self.nucleus_unet_model = nucleus_unet_model
        self.actin_unet_model = actin_unet_model
        self.unet_model_scale = unet_model_scale
        self.unet_model_thrh = unet_model_thrh
        self.unet_img_size = unet_img_size

class ImgResolution(object):

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z