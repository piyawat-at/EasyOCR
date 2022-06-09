import torch.nn as nn
# import importlib.util
# import sys
# spec = importlib.util.spec_from_file_location("feature_extraction", "trainer\modules\\feature_extraction.py")
# feature_ex = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(feature_ex)
from feature_extraction import ResNet_FeatureExtractor

class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)


    def forward(self, input, text):
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = visual_feature

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous())

        return prediction
