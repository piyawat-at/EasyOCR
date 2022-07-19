import matplotlib.pyplot as plt
from torchvision import transforms
from easyocr.utils import reformat_input
from easyocr.utils import get_image_list
from easyocr.recognition import AlignCollate
from easyocr.recognition import ListDataset
import torch.nn.functional as F
from easyocr.utils import CTCLabelConverter
import onnxruntime
import numpy as np
import torch

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()if tensor.requires_grad else tensor.cpu().numpy()
def custom_mean(x):
    return x.prod()**(2.0/np.sqrt(len(x)))

def onnx_recognize(img_path, ort_session, converter, device = torch.device("cuda")):
    img, img_cv_grey = reformat_input(img_path)
    y_max, x_max = img_cv_grey.shape
    horizontal_list = [[0, x_max, 0, y_max]]
    for bbox in horizontal_list:
                    h_list = [bbox]
                    f_list = []
                    image_list, max_width = get_image_list(h_list, f_list, img_cv_grey, model_height = 64)

    img_list = [item[1] for item in image_list]

    AlignCollate_normal = AlignCollate(imgH=64, imgW=600, keep_ratio_with_pad=True)
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False,
            num_workers=int(0), collate_fn=AlignCollate_normal, pin_memory=True)
    
    image_tensors = next(iter(test_loader))
    batch_size = image_tensors.size(0)
    image = image_tensors.to(device)



    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
    ort_outs = ort_session.run(None, ort_inputs)
    preds = torch.from_numpy(ort_outs[0])


    ignore_idx = []
    # Select max probabilty (greedy decoding) then decode index to character
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)

    ######## filter ignore_char, rebalance
    preds_prob = F.softmax(preds, dim=2)
    preds_prob = preds_prob.cpu().detach().numpy()
    preds_prob[:,:,ignore_idx] = 0.
    pred_norm = preds_prob.sum(axis=2)
    preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1)
    preds_prob = torch.from_numpy(preds_prob).float().to(device)
    result = []

    # decoder
    decoder = 'greedy'
    if decoder == 'greedy':
        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds_prob.max(2)
        preds_index = preds_index.view(-1)
        preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)
    elif decoder == 'beamsearch':
        k = preds_prob.cpu().detach().numpy()
        preds_str = converter.decode_beamsearch(k, beamWidth=5)
    elif decoder == 'wordbeamsearch':
        k = preds_prob.cpu().detach().numpy()
        preds_str = converter.decode_wordbeamsearch(k, beamWidth=5)


    preds_prob = preds_prob.cpu().detach().numpy()
    values = preds_prob.max(axis=2)
    indices = preds_prob.argmax(axis=2)
    preds_max_prob = []
    for v,i in zip(values, indices):
        max_probs = v[i!=0]
        if len(max_probs)>0:
            preds_max_prob.append(max_probs)
        else:
            preds_max_prob.append(np.array([0]))

    for pred, pred_max_prob in zip(preds_str, preds_max_prob):
        confidence_score = custom_mean(pred_max_prob)
        result.append([pred, confidence_score])
    return result