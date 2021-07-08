from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GAIN_regression(nn.Module, ABC):
    def __init__(self, model, grad_layer, num_classes):
        super(GAIN_regression, self).__init__()
        self.model = model
        # print(self.model)
        self.grad_layer = grad_layer

        self.num_classes = num_classes

        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None

        # Register hooks
        self._register_hooks(grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100

    def _to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.num_classes), requires_grad=True)
        #         print('ohe', ohe)
        for i, label in enumerate(labels):
            ohe[i, label] = 1

        ohe = torch.autograd.Variable(ohe)
        # print('ohe', ohe)
        return ohe

    def _register_hooks(self, grad_layer):
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output
            # print(grad_output.shape)

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]

            # print(grad_output[0].shape)

        gradient_layer_found = False
        for idx, m in self.model.named_modules():

            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def forward(self, images):

        is_train = self.model.training
        #         print('is_train', is_train)
        with torch.enable_grad():
            # labels_ohe = self._to_ohe(labels).cuda()
            # labels_ohe.requires_grad = True

            _, _, img_h, img_w = images.size()

            self.model.train(True)
            logits = self.model(images)  # BS x num_classes
            self.model.zero_grad()

            pred = logits.sigmoid()
            # import pdb; pdb.set_trace()
            # print('pred', pred.shape)

            # for predict in pred:
            pred = pred.sum()

            # out_logits_am = []
            # out_heatmap = []
            pred.backward(retain_graph=True)
            self.model.zero_grad()

            backward_features = self.backward_features  # BS x C x H x W
            # print(backward_features.shape)
            # bs, c, h, w = backward_features.size()
            # wc = F.avg_pool2d(backward_features, (h, w), 1)  # BS x C x 1 x 1

            """
            The wc shows how important of the features map
            """

            # Eq 2
            fl = self.feed_forward_features  # BS x C x H x W
            # print(fl.shape)
            # bs, c, h, w = fl.size()
            # fl = fl.view(1, bs * c, h, w)

            """
            fl is the feature maps during feed-forward
            """

            """
            We do 2d convolution to find the Attention maps. We consider wc as a filter matrix.
            """

            # Ac = F.relu(F.conv2d(fl, wc, groups=bs))
            # # Resize to be as same as of image size
            # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
            # Ac = Ac.permute((1, 0, 2, 3))
            # heatmap = Ac
            # import pdb; pdb.set_trace()
            weights = F.adaptive_avg_pool2d(backward_features, 1)
            Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
            Ac = F.relu(Ac)
            # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
            Ac = F.upsample_bilinear(Ac, size=images.size()[2:])
            heatmap = Ac

            """
            Generate the soft-mask
            """

            Ac_min = Ac.min()
            Ac_max = Ac.max()
            scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min)
            mask = F.sigmoid(self.omega * (scaled_ac - self.sigma))
            masked_image = images - images * mask

            logits_am = self.model(masked_image)
            # print(logits_am)
            # out_logits_am.append(logits_am)
            # out_logits_am = torch.cat(out_logits_am)
            # out_logits_am = out_logits_am.sum(dim=0)
            #
            # out_heatmap.append(heatmap)

        return logits, logits_am, heatmap


class GAIN_classification(nn.Module, ABC):
    def __init__(self, model, grad_layer, num_classes):
        super(GAIN_classification, self).__init__()
        self.model = model
        # print(self.model)
        self.grad_layer = grad_layer

        self.num_classes = num_classes

        # Feed-forward features
        self.feed_forward_features = None
        # Backward features
        self.backward_features = None

        # Register hooks
        self._register_hooks(grad_layer)

        # sigma, omega for making the soft-mask
        self.sigma = 0.25
        self.omega = 100

    def _register_hooks(self, grad_layer):
        def forward_hook(module, grad_input, grad_output):
            self.feed_forward_features = grad_output
            # print(grad_output)

        def backward_hook(module, grad_input, grad_output):
            self.backward_features = grad_output[0]
            # print(grad_output.shape)

        gradient_layer_found = False
        for idx, m in self.model.named_modules():
            if idx == grad_layer:
                m.register_forward_hook(forward_hook)
                m.register_backward_hook(backward_hook)
                print("Register forward hook !")
                print("Register backward hook !")
                gradient_layer_found = True
                break

        # for our own sanity, confirm its existence
        if not gradient_layer_found:
            raise AttributeError('Gradient layer %s not found in the internal model' % grad_layer)

    def _to_ohe(self, labels):
        ohe = torch.zeros((labels.size(0), self.num_classes), requires_grad=True)
        for i, label in enumerate(labels):
            ohe[i, label] = 1

        ohe = torch.autograd.Variable(ohe)
        # print('ohe', ohe.shape)
        return ohe

    def forward(self, images, labels):

        # Remember, only do back-probagation during the training. During the validation, it will be affected by bachnorm
        # dropout, etc. It leads to unstable validation score. It is better to visualize attention maps at the testset

        is_train = self.model.training

        with torch.enable_grad():
            # labels_ohe = self._to_ohe(labels).cuda()
            # labels_ohe.requires_grad = True

            _, _, img_h, img_w = images.size()

            self.model.train(True)
            _, logits = self.model(images)  # BS x num_classes
            self.model.zero_grad()

            if not is_train:
                pred = F.softmax(logits).argmax(dim=1)
                labels_ohe = self._to_ohe(pred).cuda()
            else:
                labels_ohe = self._to_ohe(labels).cuda()
                # print('labels_ohe', labels_ohe.shape)

            gradient = logits * labels_ohe
            grad_logits = (logits * labels_ohe).sum()  # BS x num_classes
            grad_logits.backward(retain_graph=True)
            self.model.zero_grad()

        if is_train:
            self.model.train(True)
        else:
            self.model.train(False)
            self.model.eval()
            logits = self.model(images)

        backward_features = self.backward_features  # BS x C x H x W
        # bs, c, h, w = backward_features.size()
        # wc = F.avg_pool2d(backward_features, (h, w), 1)  # BS x C x 1 x 1

        """
        The wc shows how important of the features map
        """

        # Eq 2
        fl = self.feed_forward_features  # BS x C x H x W
        # print(fl.shape)
        # bs, c, h, w = fl.size()
        # fl = fl.view(1, bs * c, h, w)

        """
        fl is the feature maps during feed-forward
        """

        """
        We do 2d convolution to find the Attention maps. We consider wc as a filter matrix.
        """

        # Ac = F.relu(F.conv2d(fl, wc, groups=bs))
        # # Resize to be as same as of image size
        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        # Ac = Ac.permute((1, 0, 2, 3))
        # heatmap = Ac

        weights = F.adaptive_avg_pool2d(backward_features, 1)
        Ac = torch.mul(fl, weights).sum(dim=1, keepdim=True)
        Ac = F.relu(Ac)
        # Ac = F.interpolate(Ac, size=images.size()[2:], mode='bilinear', align_corners=False)
        Ac = F.upsample_bilinear(Ac, size=images.size()[2:])
        heatmap = Ac

        """
        Generate the soft-mask
        """

        Ac_min = Ac.min()
        Ac_max = Ac.max()
        scaled_ac = (Ac - Ac_min) / (Ac_max - Ac_min)
        mask = F.sigmoid(self.omega * (scaled_ac - self.sigma))
        masked_image = images - images * mask

        _, logits_am = self.model(masked_image)

        return logits, logits_am, heatmap
