import torch
from models.resnet import resnet50


def prepare_model_relation(file_name='UnbiasedEmo_best'):
    print('Model EmotionNet Loaded')
    save_file = f'./weight/{file_name}.pth.tar'
    checkpoint = torch.load(save_file)

    backbone = resnet50(pretrained=True)

    new_state_dict = {}
    for key, values in checkpoint['state_dict'].items():
        new_state_dict[key.replace('module.', '')] = values

    del new_state_dict['fc.weight']
    del new_state_dict['fc.bias']

    backbone.load_state_dict(new_state_dict, strict=False)
    return backbone
