import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import torchvision
import parameters as params
import timm

class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, out_channels=256, out_size=(64, 64)):
        super().__init__()
        self.dino = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad = False

        self.out_size = out_size
        self.feat_proj = nn.Sequential(
            nn.Conv2d(self.dino.embed_dim, out_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(518, 518), mode='bilinear', align_corners=False)
        patch_tokens = self.dino.forward_features(x)
        patch_tokens = patch_tokens[:, 1:]
        B, N, C = patch_tokens.shape
        h = w = int(N ** 0.5)
        feat_map = patch_tokens.transpose(1, 2).reshape(B, C, h, w)  # [B, C, H', W']
        feat_map = F.interpolate(feat_map, size=self.out_size, mode='bilinear', align_corners=False)
        return self.feat_proj(feat_map)
    
def getLinearLayer(in_feat, out_feat, activation=nn.ReLU(True)):
    return nn.Sequential(
        nn.Linear(in_features=in_feat, out_features=out_feat, bias=True),
        activation
    )

def getConvLayer(in_channel,out_channel,stride=1,padding=1,activation=nn.ReLU()):
    return nn.Sequential(nn.Conv2d(in_channel, 
                    out_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    padding_mode='reflect'),
                    activation)

def getConvTransposeLayer(in_channel, out_channel,kernel=3,stride=1,padding=1,activation=nn.ReLU()):
    return nn.Sequential(nn.ConvTranspose2d(in_channel,
                                            out_channel,
                                            kernel_size = kernel,
                                            stride=stride,
                                            padding=padding),
                                            activation)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + self.shortcut(residual)
        out = self.relu(out)
        return out


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, expansion=4):
#         super().__init__()
#         mid_channels = out_channels // expansion
#         self.pw_reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(mid_channels)
#         self.dw = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
#                             stride=stride, padding=1, groups=mid_channels, bias=False)
#         self.bn2 = nn.BatchNorm2d(mid_channels)
#         self.pw_expand = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )
#         else:
#             self.shortcut = nn.Identity()

#     def forward(self, x):
#         identity = x

#         out = self.pw_reduce(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.dw(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.pw_expand(out)
#         out = self.bn3(out)

#         out += self.shortcut(identity)
#         out = self.relu(out)
#         return out

class FeatureNet(nn.Module):
    def __init__(self,height,width):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=False)
        layers = list(model.children())
        self.FeatureEncoder = torch.nn.Sequential(*layers[:5].copy())
        self.expand_layer = ResidualBlock(256, 200)

    def forward(self, x):
        x = self.FeatureEncoder(x)
        x = self.expand_layer(x)
        return x

    def apply_feature_encoder(self, x):
        x = self.FeatureEncoder(x)
        x = self.expand_layer(x)
        return x

class Encoder(nn.Module):
    def __init__(self, height, width, total_image_input=1):
        super().__init__()
        self.height = height
        self.width = width
        self.encoder_pre = ResidualBlock((total_image_input*3), 20)
        self.encoder_layer1 = ResidualBlock(20, 30)
        self.encoder_layer2 = ResidualBlock(30, 50)

        self.encoder_layer3 = nn.Sequential(
            ResidualBlock(50, 100),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder_layer4 = ResidualBlock(100, 200)
        self.encoder_layer5 = nn.Sequential(
            ResidualBlock(200, 200),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder_layer6 = ResidualBlock(200, 200)
        self.encoder_layer7 = nn.Sequential(
            ResidualBlock(200, 200),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder_layer8 = ResidualBlock(200, 500)
        self.encoder_layer9 = nn.Sequential(
            ResidualBlock(500, 500),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
                                
        self.encoder_layer10 = ResidualBlock(500, 500)
        self.encoder_layer11 = ResidualBlock(500, 500)
        
    def forward(self, x, height=None, width=None):
        if height == None and width == None:
            height = self.height
            width = self.width

        x = self.encoder_pre(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        skip1 = self.encoder_layer3(x)
        
        x = self.encoder_layer4(skip1)
        skip2 = self.encoder_layer5(x)

        x = self.encoder_layer6(skip2)
        skip3 = self.encoder_layer7(x)

        x = self.encoder_layer8(skip3)
        skip4 = self.encoder_layer9(x)

        x = self.encoder_layer10(skip4)
        x = self.encoder_layer11(x)

        return x, [skip1, skip2, skip3, skip4]

class DecoderRGB(nn.Module):
    def __init__(self,height,width):
        super().__init__()
        self.height = height
        self.width = width
        self.decoder_layer1 = ResidualBlock(500, 500)
        self.decoder_layer2 = ResidualBlock(500, 500)
        self.decoder_layer3 = ResidualBlock(500, 500)
        
        self.decoder_layer4 = nn.Sequential(
            nn.ConvTranspose2d(500, 200, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer5 = ResidualBlock(200, 200)
    
        self.decoder_layer6 = nn.Sequential(
            nn.ConvTranspose2d(200, 200, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer7 = ResidualBlock(200, 200)

        self.decoder_layer8 = nn.Sequential(
            nn.ConvTranspose2d(200, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer9 = ResidualBlock(100, 100)

        self.decoder_layer10 = nn.Sequential(
            nn.ConvTranspose2d(100, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer11 = ResidualBlock(100, 100)
        self.decoder_layer12 = ResidualBlock(100, 96)
        self.decoder_layer13 = ResidualBlock(96, 96)
        self.decoder_layer14 = ResidualBlock(96, 96)
        self.decoder_layer15 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.decoder_layer16 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, lower_skip_list, imagenet_features, height=None, width=None):
        if height == None and width == None:
            height = self.height
            width = self.width

        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = x + lower_skip_list[3]

        x = self.decoder_layer3(x)
        x = self.decoder_layer4(x)
        x = x + lower_skip_list[2]
        
        x = self.decoder_layer5(x)
        x = self.decoder_layer6(x)
        x = x + lower_skip_list[1] + imagenet_features

        x = self.decoder_layer7(x)
        x = self.decoder_layer8(x)
        x = x + lower_skip_list[0]

        x = self.decoder_layer9(x)
        x = self.decoder_layer10(x)
        x = self.decoder_layer11(x)
        x = self.decoder_layer12(x)
        x = self.decoder_layer13(x)
        x = self.decoder_layer14(x)
        x = self.decoder_layer15(x)
        x = self.decoder_layer16(x)       
        x = x.view(x.size()[0], 32, 3, height, width)
        return x
    
class DecoderSigma(nn.Module):
    def __init__(self,height,width):
        super().__init__()
        self.height = height
        self.width = width
        self.decoder_layer1 = ResidualBlock(500, 500)
        self.decoder_layer2 = ResidualBlock(500, 500)
        self.decoder_layer3 = ResidualBlock(500, 500)
        
        self.decoder_layer4 = nn.Sequential(
            nn.ConvTranspose2d(500, 200, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer5 = ResidualBlock(200, 200)
    
        self.decoder_layer6 = nn.Sequential(
            nn.ConvTranspose2d(200, 200, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer7 = ResidualBlock(200, 200)

        self.decoder_layer8 = nn.Sequential(
            nn.ConvTranspose2d(200, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer9 = ResidualBlock(100, 100)

        self.decoder_layer10 = nn.Sequential(
            nn.ConvTranspose2d(100, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer11 = ResidualBlock(100, 100)
        self.decoder_layer12 = ResidualBlock(100, 50)
        self.decoder_layer13 = ResidualBlock(50, 40)
        self.decoder_layer14 = ResidualBlock(40, 32)
        self.decoder_layer15 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.decoder_layer16 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x, lower_skip_list, imagenet_features, height=None, width=None):
        if height == None and width == None:
            height = self.height
            width = self.width

        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = x + lower_skip_list[3]

        x = self.decoder_layer3(x)
        x = self.decoder_layer4(x)
        x = x + lower_skip_list[2]
        
        x = self.decoder_layer5(x)
        x = self.decoder_layer6(x)
        x = x + lower_skip_list[1] + imagenet_features

        x = self.decoder_layer7(x)
        x = self.decoder_layer8(x)
        x = x + lower_skip_list[0]

        x = self.decoder_layer9(x)
        x = self.decoder_layer10(x)
        x = self.decoder_layer11(x)
        x = self.decoder_layer12(x)
        x = self.decoder_layer13(x)
        x = self.decoder_layer14(x)
        x = self.decoder_layer15(x)
        x = self.decoder_layer16(x)          
        x = x.view(x.size()[0], 32, 1, height, width)
        return x
    

class DecoderDepth(nn.Module):
    def __init__(self,height,width):
        super().__init__()
        self.height = height
        self.width = width
        self.decoder_layer1 = ResidualBlock(500, 500)
        self.decoder_layer2 = ResidualBlock(500, 500)
        self.decoder_layer3 = ResidualBlock(500, 500)
        
        self.decoder_layer4 = nn.Sequential(
            nn.ConvTranspose2d(500, 200, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer5 = ResidualBlock(200, 200)
    
        self.decoder_layer6 = nn.Sequential(
            nn.ConvTranspose2d(200, 200, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer7 = ResidualBlock(200, 200)

        self.decoder_layer8 = nn.Sequential(
            nn.ConvTranspose2d(200, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer9 = ResidualBlock(100, 100)

        self.decoder_layer10 = nn.Sequential(
            nn.ConvTranspose2d(100, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer11 = ResidualBlock(100, 100)
        self.decoder_layer12 = ResidualBlock(100, 50)
        self.decoder_layer13 = ResidualBlock(50, 40)
        self.decoder_layer14 = ResidualBlock(40, 16)
        self.decoder_layer15 = nn.Sequential(
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.decoder_layer16 = nn.Sequential(
            nn.Conv2d(8, 1, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x, lower_skip_list, imagenet_features, height=None, width=None):
        if height == None and width == None:
            height = self.height
            width = self.width

        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = x + lower_skip_list[3]

        x = self.decoder_layer3(x)
        x = self.decoder_layer4(x)
        x = x + lower_skip_list[2]
        
        x = self.decoder_layer5(x)
        x = self.decoder_layer6(x)
        x = x + lower_skip_list[1] + imagenet_features

        x = self.decoder_layer7(x)
        x = self.decoder_layer8(x)
        x = x + lower_skip_list[0]

        x = self.decoder_layer9(x)
        x = self.decoder_layer10(x)
        x = self.decoder_layer11(x)
        x = self.decoder_layer12(x)
        x = self.decoder_layer13(x)
        x = self.decoder_layer14(x)
        x = self.decoder_layer15(x)
        x = self.decoder_layer16(x)
        return x

class MMPI(nn.Module):
    def __init__(self,total_image_input=1, height=384,width=384):
        super().__init__()
        self.height = height
        self.width = width
        self.feature_encoder = FeatureNet(height,width)
        self.lower_encoder = Encoder(height, width, total_image_input)
        self.merge_decoder_rgb = DecoderRGB(height, width)
        self.merge_decoder_sigma = DecoderSigma(height, width)
        self.depth_decoder = DecoderDepth(height, width)
        
    def forward(self, x, height=None, width=None):
        if height == None and width == None:
            height = self.height
            width = self.width

        imagenet_fatures = self.feature_encoder.apply_feature_encoder(x)
        lower_feature, skip_list = self.lower_encoder(x, height, width)

        merged_feature_rgb = self.merge_decoder_rgb(lower_feature, skip_list, imagenet_fatures, height, width)
        merged_feature_sigma = self.merge_decoder_sigma(lower_feature, skip_list, imagenet_fatures, height, width)

        merged_feature_depth = self.depth_decoder(lower_feature, skip_list, imagenet_fatures)

        return merged_feature_rgb, merged_feature_sigma, merged_feature_depth
        
    def get_rgb_sigma(self, x, height=None, width=None):
        if height == None and width == None:
            height = self.height
            width = self.width

        imagenet_fatures = self.feature_encoder.apply_feature_encoder(x)
        lower_feature, skip_list = self.lower_encoder(x, height, width)
        merged_feature_rgb = self.merge_decoder_rgb(lower_feature, skip_list, imagenet_fatures, height, width)
        merged_feature_sigma = self.merge_decoder_sigma(lower_feature, skip_list, imagenet_fatures, height, width)
        return merged_feature_rgb, merged_feature_sigma

    def get_depth(self, x, height=None, width=None):
        if height == None and width == None:
            height = self.height
            width = self.width

        imagenet_fatures = self.feature_encoder.apply_feature_encoder(x)
        lower_feature, skip_list = self.lower_encoder(x, height, width)
        merged_feature_depth = self.depth_decoder(lower_feature, skip_list, imagenet_fatures)
        return merged_feature_depth

    def get_layer_depth(self, x, grid, height=None, width=None):
        if height == None and width == None:
            height = self.height
            width = self.width

        imagenet_fatures = self.feature_encoder.apply_feature_encoder(x)
        lower_feature, skip_list = self.lower_encoder(x, height, width)

        rgb_layers = self.merge_decoder_rgb(lower_feature, skip_list, imagenet_fatures, height, width)
        sigma_layers = self.merge_decoder_sigma(lower_feature, skip_list, imagenet_fatures, height, width)

        pred_mpi_planes = torch.randn((1, 4, height, width)).to(params.DEVICE)
        for i in range(params.params_num_planes):
            RGBA = torch.cat((rgb_layers[0,i,:,:,:],sigma_layers[0,i,:,:,:]),dim=0).unsqueeze(0)
            pred_mpi_planes = torch.cat((pred_mpi_planes,RGBA),dim=0)
        
        pred_mpi_planes = pred_mpi_planes[1:,:,:,:].unsqueeze(0)
        
        sigma = pred_mpi_planes[:, :, 3, :, :]
        B, D, H, W = sigma.shape

        pred_mpi_disp = grid
        disp_sorted, _ = pred_mpi_disp.sort(dim=1)
        delta = disp_sorted[:, 1:] - disp_sorted[:, :-1]
        delta_last = delta[:, -1:]
        delta = torch.cat([delta, delta_last], dim=1)

        delta = delta.unsqueeze(-1).unsqueeze(-1).expand_as(sigma)

        alpha = 1.0 - torch.exp(-delta * sigma)

        transmittance = torch.cumprod(1 - alpha + 1e-7, dim=1)
        shifted_transmittance = torch.ones_like(transmittance)
        shifted_transmittance[:, 1:, :, :] = transmittance[:, :-1, :, :]

        disparity = pred_mpi_disp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

        disparity_map = (disparity * alpha * shifted_transmittance).sum(dim=1, keepdim=True)

        return disparity_map

    def get_layers(self, x, height=None, width=None):
        if height == None and width == None:
            height = self.height
            width = self.width

        imagenet_fatures = self.feature_encoder.apply_feature_encoder(x)
        lower_feature, skip_list = self.lower_encoder(x, height, width)
        merged_feature_rgb = self.merge_decoder_rgb(lower_feature, skip_list, imagenet_fatures, height, width)
        merged_feature_sigma = self.merge_decoder_sigma(lower_feature, skip_list, imagenet_fatures, height, width)
        return merged_feature_rgb, merged_feature_sigma
        


        
    
    
    
