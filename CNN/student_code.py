import segmentation_models_pytorch as smp
import albumentations as albu
import torch
import utils as utils
import torchinfo as torchinfo
import os
from torchvision.io import read_image
import numpy as np


DEVICE = 'cpu'


# def apply_softmax(image, mask):
# TO-DO 1
def sigmoid(x):
    '''
    Implement the sigmoid function using the formula given in the proj2 notebook. 
    You are not allowed to use any pre-defined sigmoid function from any library. 
    
    Args:
        x: N x 1 torch.FloatTensor
        
    Returns:
        y: N x 1 torch.FloatTensor
    '''
    y = None
    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    
    y = torch.exp(- 1 * x)
    y = torch.add(y, 1)
    y = torch.reciprocal(y)


    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return y

# TO-DO 2
def softmax(x):
    '''
    Implement the softmax function using the formula given in the proj2 notebook. 
    You are not allowed to use any pre-defined softmax function from any library. 
    
    Args:
        x: N x 1 torch.FloatTensor
        
    Returns:
        y: N x 1 torch.FloatTensor
    '''
    y = None
    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    
    x = torch.exp(x)
    sum = torch.sum(x)
    y = torch.div(x, sum)

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return y

# TO-DO 3
def apply_mask_to_image(image, mask):
    '''
    Applies segmentation mask of image to the original image to produce the final segmented image. 
    
    For this function, we will be applying the color red to the final segmented image.
    
    You will need to implement these steps:
    1.) Make `mask` the same shape as `image` such that each channel in the resulting `colored_mask` is equal to `mask`.
    2.) Perform broadcast multiplication on `mask` with the red color list in the `colors` dictionary. 
        The resulting `colored_mask` should contain values different from 0 in the red channel of the mask ONLY. 
    3.) Combine `image` and `colored_mask` by performing element-wise summation, and return the result as type integer.
    4.) Make sure all values in the `final_seg_img` fall between 0 and 255.
    
    '''
    image = torch.from_numpy(image).type(torch.int64)
    mask = torch.from_numpy(mask).type(torch.int64)
    colors = {"red": torch.Tensor([255,0,0])}
    colored_mask = None
    final_seg_img = None
    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    '''
    if len(mask.shape) == 2:
        mask = torch.unsqueeze(2)
    '''
    mask = mask.repeat(1,1,image.shape[2])
    colored_mask = mask * colors['red']
    final_seg_img = image + colored_mask
    final_seg_img = torch.clamp(final_seg_img, min = 0, max = 255)
    final_seg_img = final_seg_img.type(torch.int64)

    #print("mask",mask.size())
    #print("img",image.size())


    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    
    return final_seg_img


# TO-DO 4
def load_FPN_resnet50():
    '''
    Repeat the steps at the beginning of Section 2.2: VGG-19 for the ResNet-50 model to see 
    how well it performs in the cell below. 
    
    You will keep the same `classes`, and `activation` function. But, you will need to change 
    the `encoder_name` to `"resnet50"` to load the ResNet-50 model from `segmentation_models_pytorch`. 
    
    Please see the `models/` directory for the appropriate saved weights file for the ResNet-50 model.
    '''
    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    resnet50_model = smp.FPN(
        encoder_name='resnet50', 
        classes=1, 
        activation='sigmoid',
    )
    resnet50_model.load_state_dict(torch.load('./models/resnet50_best_model_weights.pt', map_location=torch.device('cpu')))

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return resnet50_model

# TO-DO 5
def IoU(predict: torch.Tensor, target: torch.Tensor):

    """Compute IoU on torch tensor
    
    Args:
        predict: MxN torch tensor represeting predicted label map,
            each value in range 0 to 1.
        target: MxN torch tensor representing ground truth label map,
            each value in range 0 to 1.

    Returns: Float/Double that represent the IoU score
    """
    
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    IoU = 1.0 * torch.sum(torch.logical_and(predict, target)) / torch.sum(torch.logical_or(predict, target))
    #print("overlap sum", torch.sum(torch.logical_and(predict, target)))
    #print("union sum", torch.sum(torch.logical_or(predict, target)))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return IoU

# TO-DO 6
def applyIoU(model: smp.fpn.model, dataset: utils.Dataset):
    
    """
    Apply the IoU function you wrote above to evaluate the performance 
    of the input model on the input dataset.
    
    Hint: true_mask is a np.ndarray. You will need to convert it to torch.Tensor; 
    otherwise, you will get an AssertionError.
    
    Args: 
        model: the smp model that you will evaluate
        dataset: the dataset that model will be tested on; 
                 It contains N (image, true mask) pair
    Returns:
        IoU_score: N Iou scores, one score corresponding to one pair
        
    """
    
    IoU_score = []
    
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    for i in range(len(dataset)):
        image, true_mask = dataset[i]
        img_tensor = torch.from_numpy(image).to('cpu').unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).type(torch.int64)
        predict_mask = model.predict(img_tensor)
        predict_mask = predict_mask.squeeze()
        predict_mask = torch.round(predict_mask)
        predict_mask = predict_mask.type(torch.int64)
        true_mask = true_mask.squeeze()
        IoU_score.append(IoU(predict_mask, true_mask))
        #print("predict size", predict_mask.size())
        #print("true mask size", true_mask.size())
        #print("predict", predict_mask)
        #print("true mask", true_mask)

       
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    assert type(true_mask) == torch.Tensor
    
    return IoU_score

# TO-DO 7
def compare_psp_fpn(test_dataset):
    """
    Step 1: get FPN segmentation model with resnet50 from above
    Step 2: create PSP segmentation model with pretrained resnet50 for PSP
    Step 3: Get the IoU score 
    
    Args:
        test_dataset: the dataset that we will use to find out IoU score
        
    Returns:
        psp_iou: the IoU score for the PSPNet with Resnet50 
        fpn_iou: the IoU score for the FPN with Resnet50
        
    
    """
    
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    resnet50_model = load_FPN_resnet50()
    pspresnet50_model = smp.PSPNet(
        encoder_name='resnet50', 
        classes=1, 
        activation='sigmoid',
    )
    pspresnet50_model.load_state_dict(torch.load('./models/pspnet_resnet50_best_model_weights.pt', map_location=torch.device('cpu')))

    psp_iou = applyIoU(pspresnet50_model, test_dataset)
    fpn_iou = applyIoU(resnet50_model, test_dataset)

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return psp_iou, fpn_iou

def load_model(decoder_weights_path=None):
    ENCODER = 'resnet50' 
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['car']
    ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation

    # create segmentation model with pretrained encoder
    model = smp.FPN(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
    )

    model.load_state_dict(torch.load('models/{}_best_model_weights.pt'.format(ENCODER), map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load(decoder_weights_path, map_location=torch.device('cpu')))
    return model

def print_model_summary(model, channels=3, H=384, W=480):
    print(torchinfo.summary(model, input_size=(1, channels, H, W)))
    

def create_vis_dataset():
    # should paths and classes be defined here or in notebook?
    x_vis_dir = "./data/CamVid/test/"
    y_vis_dir = "./data/CamVid/testannot/"
    classes = ["car"]
    return utils.Dataset(x_vis_dir, y_vis_dir, classes)

def create_test_dataset():

    x_test_dir = "./data/CamVid/test/"
    y_test_dir = "./data/CamVid/testannot/"
    classes = ["car"]
    ENCODER = 'resnet50' 
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    test_dataset = utils.Dataset(
        x_test_dir, 
        y_test_dir, 
        augmentation=utils.get_validation_augmentation(), 
        preprocessing=utils.get_preprocessing(preprocessing_fn),
        classes=classes,
    )

    return test_dataset

def test_model(model, test_dataset, vis_dataset):

    for i in range(len(test_dataset)):

        image_vis = vis_dataset[i][0].astype('uint8')
        image, gt_mask = test_dataset[i]
        
        gt_mask = gt_mask.squeeze()
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        print(x_tensor.shape)
        pr_mask = model.predict(x_tensor)
        print(f"Shape of predicted mask = {pr_mask.shape}")
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            
        utils.visualize(
            image=image_vis, 
            ground_truth_mask=gt_mask, 
            predicted_mask=pr_mask
        )
    return image, pr_mask
