import torch

from python_research.experiments.hsi_attention.models.util import build_convolutional_block, AttentionBlock, \
    build_classifier_block, \
    build_classifier_confidence


class Model3(torch.nn.Module):

    def __init__(self, num_of_classes, input_dimension, uses_attention: bool = False):
        super(Model3, self).__init__()
        self._conv_block_1 = build_convolutional_block(1, 96)
        self._conv_block_2 = build_convolutional_block(96, 54)
        self._conv_block_3 = build_convolutional_block(54, 36)
        self._attention_block_1 = AttentionBlock(96, int(input_dimension / 2), num_of_classes)
        self._attention_block_2 = AttentionBlock(54, int(input_dimension / 4), num_of_classes)
        self._attention_block_3 = AttentionBlock(36, int(input_dimension / 8), num_of_classes)
        self._classifier = build_classifier_block(36 * int(input_dimension / 8), num_of_classes)
        self._classifier_confidence = build_classifier_confidence(36 * int(input_dimension / 8))
        self._optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self._loss = torch.nn.CrossEntropyLoss()
        self._uses_attention = uses_attention

    def with_attention(self, on):
        self._uses_attention = on

    def forward(self, x):
        z = self._conv_block_1(x)
        if self._uses_attention:
            heatmap_1 = self._attention_block_1(z)
        z = self._conv_block_2(z)
        if self._uses_attention:
            heatmap_2 = self._attention_block_2(z)
        z = self._conv_block_3(z)
        if self._uses_attention:
            heatmap_3 = self._attention_block_3(z)
        prediction = self._classifier(z.view(z.shape[0], -1) * self._classifier_confidence(z.view(z.shape[0], -1)))
        if self._uses_attention:
            return prediction + heatmap_1 + heatmap_2 + heatmap_3
        return prediction

    def get_heatmaps(self, input_size):
        return [self._attention_block_1.get_heatmaps(input_size), self._attention_block_2.get_heatmaps(input_size),
                self._attention_block_3.get_heatmaps(input_size)]