from torch import nn
import time
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified, Normalize, SPDNormalization, SPDTransform_scaled, SCaled_graph, SCaled_weighted_graph
from mean_shift import GBMS_RNN, SPD_GBMS_RNN
from utils import fc2vector

class MSNet(nn.Module):
    def __init__(self, num_classes):
        super(MSNet, self).__init__()
        self.layers = nn.Sequential(

            # SPDTransform(58, 48, 1),
            # SPDRectified(),
            # SPDTransform(48, 36, 1),
            # SPDRectified(),
            # SPDTransform(36, 24, 1),
            # SPDRectified(),
            # SPDTransform(24, 16, 1),
            # SPDTangentSpace(24, vectorize_all=False),
            # Normalize(),

            # whole brain
            # SPDTransform(268, 128, 1),
            # SPDRectified(),
            # SPDTransform(128, 64, 1),
            # SPDRectified(),
            # SPDTransform(64, 32,  1),
            # SPDRectified(),
            # SPDTransform(32, 16, 1),
            # Normalize(),

            # simulated data
            SPDTransform(10, 6, 1),
            SPDRectified(),
            SPDTransform(6, 4, 1),
            SPDRectified(),
            SPDTransform(4, 2, 1),
            SPDTangentSpace(),
            Normalize(),


            # SPDTransform(116, 64, 1),
            # SPDRectified(),
            # SPDTransform(64, 32, 1),
            # SPDRectified(),
            # SPDTransform(32, 16, 1),
            # SPDTangentSpace(vectorize_all=False),
            # Normalize(),

            # GBMS_RNN(normalize=True),
            # GBMS_RNN(normalize=True),
            # GBMS_RNN(normalize=True)
        )
        self.classifier = nn.Sequential(
            # nn.Linear(256,16 * 17 // 2),
            # nn.Linear(16 * 17 // 2, num_classes),
            nn.Linear(2*3 // 2, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        out = self.classifier(x)
        return x, out

class MSNet_predict(nn.Module):
    def __init__(self, num_classes):
        super(MSNet_predict, self).__init__()
        self.layers = nn.Sequential(

            # real data
            # SPDNormalization(58),
            # SPDTransform(58, 48, 1),
            # SPDNormalization(48),
            # SPDRectified(),
            # SPDTransform(48, 36, 1),
            # SPDNormalization(36),
            # SPDRectified(),
            # SPDTransform(36, 24, 1),
            # SPDNormalization(24),
            # SPDRectified(),
            # SPDTransform(24,16, 1),
            # SPDNormalization(16),

            # simulated data
            # SPDTransform(10, 9, 1),
            # SPDRectified(),
            # SPDTransform(9, 8, 1),
            # SPDRectified(),
            # SPDTransform(8, 6, 1),
            SPDTransform(116, 100, 1),
            SPDRectified(),
            SPDTransform(100, 90, 1),
            # SPDRectified(),
            # SPDTransform(32, 16, 1),
            # SPD_GBMS_RNN(),
            # SPD_GBMS_RNN(),
            # SPD_GBMS_RNN()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class SPDNet(nn.Module):
    def __init__(self, num_classes):
        super(SPDNet, self).__init__()
        self.layers = nn.Sequential(

            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDRectified(),
            SPDTransform(32, 16, 1),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier = nn.Sequential(
            # nn.Linear(256,16 * 17 // 2),
            # nn.Linear(16 * 17 // 2, num_classes),
            # nn.Linear(6*7 // 2, num_classes),
            # nn.Linear(116*115 // 2, 1280),
            # nn.Linear(1280, 640),
            # nn.Linear(640, 320),
            # nn.Linear(320, 16 * 17 // 2),
            nn.Linear(16 * 17 // 2, num_classes),
        )
        self.layers0 = nn.Sequential(

            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier0 = nn.Sequential(

            nn.Linear(32 * 33 // 2, num_classes),

        )
        self.layers1 = nn.Sequential(

            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier1 = nn.Sequential(

            nn.Linear(64 * 65 // 2, num_classes),

        )
        self.layers2 = nn.Sequential(

            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDRectified(),
            SPDTransform(32, 16, 1),
            SPDRectified(),
            SPDTransform(16, 8, 1),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier2 = nn.Sequential(

            nn.Linear(8 * 9 // 2, num_classes),

        )
        self.layers3 = nn.Sequential(

            SPDTransform(116, 16, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier3 = nn.Sequential(

            nn.Linear(16 * 17 // 2, num_classes),

        )
        self.layers4 = nn.Sequential(

            SPDTransform(116, 32, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier4 = nn.Sequential(

            nn.Linear(32 * 33 // 2, num_classes),

        )

    def forward(self, x, spd):
        # x = self.layers(x)
        # x = fc2vector(x)
        # out = self.classifier(x)
        if spd ==0:
            x = self.layers0(x)
            out = self.classifier0(x)
        elif spd ==1:
            x = self.layers1(x)
            out = self.classifier1(x)
        elif spd ==2:
            x = self.layers2(x)
            out = self.classifier2(x)
        elif spd ==3:
            x = self.layers3(x)
            out = self.classifier3(x)
        elif spd ==4:
            x = self.layers4(x)
            out = self.classifier4(x)
        else:
            x = self.layers(x)
            out = self.classifier(x)
        return x, out
    

class DeepHoloBrain(nn.Module):
    def __init__(self, num_classes):
        super(DeepHoloBrain, self).__init__()
        self.layers = nn.Sequential(

            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDRectified(),
            SPDTransform(32, 16, 1),
            SPDTangentSpace(vectorize_all=False),
            Normalize(),
        )
        self.scaled = SCaled_graph()
        self.weighted_scaled = SCaled_weighted_graph()
        self.tans = SPDTransform(116, 64, 1),
        self.rece = SPDRectified()
        self.tangentspace=SPDTangentSpace(vectorize_all=False)
        self.Nor=Normalize()
        self.classifier1 = nn.Sequential(
            # nn.Linear(116*115 // 2, 1280),
            # nn.Linear(1280, 640),
            # nn.Linear(640, 320),
            # nn.Linear(320, 16 * 17 // 2),
            nn.Linear(16 * 17 // 2, num_classes),
            # nn.Linear(6*7 // 2, num_classes),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(116*115 // 2, 1280),
            nn.Linear(1280, 640),
            nn.Linear(640, 320),
            nn.Linear(320, 16 * 17 // 2),
            nn.Linear(16 * 17 // 2, num_classes),
            # nn.Linear(6*7 // 2, num_classes),
        )

    def forward(self, x, sc, spd):
        # print(sc)
        # x, scale = self.scaled(sc)
        start_time = time.time()
        x, scale, row, colum = self.weighted_scaled(sc,spd)
        end_time = time.time()
        forward_time = end_time - start_time
        print("Forward time: {:.6f} seconds".format(forward_time))
        # x = self.tans(x)
        # x = self.rece(x)
        # start_time = time.time()
        # x = self.tangentspace(x)
        # end_time = time.time()
        # tangentspace_time = end_time - start_time
        # print("Tangent_time: {:.6f} seconds".format(forward_time))
        # x = self.Nor(x)
        # print(x.shape)
        # spd=1
        if spd==1:
            x = self.layers(x)
            out = self.classifier1(x)
        elif spd==0:
            out = self.classifier2(x)

        return scale, out, row, colum


class DeepHoloBrain_pre(nn.Module):
    def __init__(self, num_classes):
        super(DeepHoloBrain_pre, self).__init__()
        self.layers = nn.Sequential(

            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDRectified(),
            SPDTransform(32, 16, 1),
            SPDTangentSpace(vectorize_all=False),
            Normalize(),
        )
        self.scaled = SCaled_graph()
        self.weighted_scaled = SCaled_weighted_graph()
        self.tans = SPDTransform(116, 64, 1),
        self.rece = SPDRectified()
        self.tangentspace=SPDTangentSpace(vectorize_all=False)
        self.Nor=Normalize()
        self.classifier1 = nn.Sequential(
            # nn.Linear(116*115 // 2, 1280),
            # nn.Linear(1280, 640),
            # nn.Linear(640, 320),
            # nn.Linear(320, 16 * 17 // 2),
            nn.Linear(16 * 17 // 2, num_classes),
            # nn.Linear(6*7 // 2, num_classes),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(116*115 // 2, 1280),
            nn.Linear(1280, 640),
            nn.Linear(640, 320),
            nn.Linear(320, 16 * 17 // 2),
            nn.Linear(16 * 17 // 2, num_classes),
            # nn.Linear(6*7 // 2, num_classes),
        )

    def forward(self, x, sc, spd):
        # print(sc)
        # x, scale = self.scaled(sc)
        start_time = time.time()
        x, scale, row, colum = self.weighted_scaled(sc,spd)
        end_time = time.time()
        forward_time = end_time - start_time
        print("Forward time: {:.6f} seconds".format(forward_time))
        # x = self.tans(x)
        # x = self.rece(x)
        # start_time = time.time()
        # x = self.tangentspace(x)
        # end_time = time.time()
        # tangentspace_time = end_time - start_time
        # print("Tangent_time: {:.6f} seconds".format(forward_time))
        # x = self.Nor(x)
        # print(x.shape)
        # spd=1
        if spd==1:
            x = self.layers(x)
            out = self.classifier1(x)
        elif spd==0:
            out = self.classifier2(x)

        return scale, out, row, colum