import configparser
import os
import json

class Config:
    def __init__(self, filename='config.ini'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_file = os.path.join(dir_path, filename)
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.base_dir = os.path.expanduser(self.config['DEFAULT']['BaseDir'])

    def _full_path(self, path):
        # This method can be used if you decide to store relative paths in the config
        return os.path.join(self.base_dir, path) if not os.path.isabs(path) else path

    def set_environment(self):
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.config['DEFAULT']['CudaVisibleDevices']
        pass

    @property
    def comment_generation_dataset(self):
        return {
            'train': self._full_path(self.config['CommentGenerationDataset']['TrainSet']),
            'test': self._full_path(self.config['CommentGenerationDataset']['TestSet']),
            'validation': self._full_path(self.config['CommentGenerationDataset']['ValidationSet']),
        }

    @property
    def code_refinement_dataset(self):
        return {
            'train': self._full_path(self.config['CodeRefinementDataset']['TrainSet']),
            'test': self._full_path(self.config['CodeRefinementDataset']['TestSet']),
            'validation': self._full_path(self.config['CodeRefinementDataset']['ValidationSet']),
        }

    @property
    def crystal_bleu(self):
        return {
            'trivial_ngrams': self._full_path(self.config['CrystalBLEU']['TrivialNgrams']),
        }
    
    # def __str__(self):
    #     c = {
    #         'BaseDir': self.base_dir,
    #         'CommentGenerationDataset': self.comment_generation_dataset,
    #         'CodeRefinementDataset': self.code_refinement_dataset,
    #         'CrystalBLEU': self.crystal_bleu,
    #     }
    #     return json.dumps(c, indent=2)

    def __str__(self):
        # Automatically fetch all attributes including properties
        properties = {attr: getattr(self, attr) for attr in dir(self)
                        if isinstance(getattr(self, attr, None), property) or not attr.startswith('_')}
        return json.dumps(properties, default=lambda o: str(o), indent=2)

if __name__ == '__main__':
    config = Config()
    print(config)