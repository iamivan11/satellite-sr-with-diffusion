import os
import torch

class BaseModel():
    """
    Base class for models, handles device management (CUDA, MPS, CPU)
    and basic model state.
    """
    def __init__(self, opt):
        self.opt = opt
        # Device selection logic
        if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
            self.device = torch.device('cpu')
        elif opt.get('gpu_ids') is not None and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        """Receives input data for the model."""
        self.data = self.set_device(data)

    def set_device(self, data):
        """Moves data to the selected device."""
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.device)
        elif isinstance(data, list):
            for i, v in enumerate(data):
                if isinstance(v, torch.Tensor):
                    data[i] = v.to(self.device)
        else:
             data = data.to(self.device)
        return data

    def get_current_log(self):
        """Returns the current log dictionary."""
        return {}

    def get_current_visuals(self, **kwargs):
        """Returns the current visuals for logging and saving."""
        return {}

    def print_network(self):
        """Prints the network description."""
        pass

    def save_network(self, epoch, iter_step):
        """Saves the network to disk."""
        pass

    def load_network(self):
        """Loads the network from disk."""
        pass

    def get_network_description(self, network):
        """Returns a string description and parameter count of a network."""
        if isinstance(network, torch.nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n 