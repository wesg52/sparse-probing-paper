import torch
import transformer_lens


class LayerCachedModel:
    '''
    Wrapper class for TransformerLens model to enable running a model as
    for each layer for each batch get_activations() rather than
    for each batch for each layer get_activations()
    '''

    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

        self.current_layer = -1
        self.cached_residual_stream = None

    def get_activations(self, layer):
        assert layer >= self.current_layer

        # TODO: run model from [current_layer] to [layer]  while returning the activations
        # You may need to convert to float16 to avoid memory issues

        self.current_layer = layer


# TODO: test correctness (cached model prediction == regular model prediction)
# TODO: test speed of caching vs. redoing computation
