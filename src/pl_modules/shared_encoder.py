class EncoderUnet(pl.LightningModule):
    def __init__(self,
                 feature_list: list = None,
                 conv_block: nn.Module = models.DoubleConv,
                 down_conv: nn.Module = models.Down,
                 down_layer: str = "maxpool",
                 factor: int = 1,
                 drop: list or int = None):
        super().__init__()
        self.feature_list = feature_list
        self.conv = conv_block
        self.down_conv = down_conv
        self.down_layer = down_layer
        self.factor = factor
        self.drop = drop
        self.down_layers = []

        # Encoder path (downsampling path)
        for i in range(len(self.feature_list)-1):
            if i+1 == len(self.feature_list)-1:
                # For the last downsampling layer, need to include factor depending on bilinear flag
                self.down_layers.append(self.down_conv(in_channels=self.feature_list[i],
                                                       out_channels=self.feature_list[i +
                                                                                      1]//self.factor,
                                                       down_layer=self.down_layer,
                                                       drop=self.drop[i + 1]))
            else:
                self.down_layers.append(self.down_conv(in_channels=self.feature_list[i],
                                                       out_channels=self.feature_list[i+1],
                                                       down_layer=self.down_layer,
                                                       drop=self.drop[i+1]))
        self.down_layers = nn.Sequential(*self.down_layers)

    def forward(self, x):
        x = self.conv(x)
        outputs = []
        for layer in self.down_layers:
            outputs.append(x)
            x = layer(x)
        return x, outputs


class MergeUnet(pl.LightningModule):
    def __init__(self, feature_list: list = None,
                 factor: int = 1):
        super().__init__()
        self.feature_list = feature_list
        self.conv = nn.Conv2d
        self.factor = factor
        self.merge_layers = []

        # Merge layers
        for i in range(1, len(self.feature_list)):
            self.merge_layers.append(self.conv(self.feature_list[i],
                                               self.feature_list[i]//2,
                                               kernel_size=(3,3),
                                               padding='same').cuda())
        self.merge_x = self.conv(self.feature_list[-1]*2//self.factor,
                                 self.feature_list[-2],
                                 kernel_size=(3,3),
                                 padding='same').cuda()

    def forward(self, x, outputs):
        for i, (output, layer) in enumerate(zip(outputs, self.merge_layers)):
            outputs[i] = layer(output)
        x = self.merge_x(x)
        return x, outputs


class DecoderUnet(pl.LightningModule):
    def __init__(self, feature_list: list = None,
                 up_conv: nn.Module = models.Up,
                 bilinear: bool = False,
                 deterministic_up: bool = False,
                 factor: int = 1,
                 drop_flipped: list or int = None):
        super().__init__()
        self.feature_list = feature_list
        self.feature_list_flipped = np.flip(self.feature_list)
        self.up_conv = up_conv
        self.bilinear = bilinear
        self.deterministic_up = deterministic_up
        self.factor = factor
        self.drop_flipped = drop_flipped
        self.up_layers = []

        # Decoder path (Upsampling path)
        for i in range(len(self.feature_list)-1):
            if i + 1 == len(self.feature_list_flipped)-1:
                self.up_layers.append(self.up_conv(in_channels=self.feature_list_flipped[i],
                                                   out_channels=(
                                                       self.feature_list_flipped[i + 1]),
                                                   bilinear=self.bilinear,
                                                   deterministic=self.deterministic_up,
                                                   drop=self.drop_flipped[i]))
            else:
                self.up_layers.append(self.up_conv(in_channels=self.feature_list_flipped[i],
                                                   out_channels=(
                                                       self.feature_list_flipped[i+1]//self.factor),
                                                   bilinear=self.bilinear,
                                                   deterministic=self.deterministic_up,
                                                   drop=self.drop_flipped[i]))
        self.up_layers = nn.Sequential(*self.up_layers)

    def forward(self, x, outputs):
        for i, layer in enumerate(self.up_layers):
            x = layer(x, outputs[-(i + 1)])
        return x