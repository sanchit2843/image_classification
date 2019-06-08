import torch
from torch.autograd import Variable
from torchvision import transforms
from utils import img_plot

def predict(image,device,encoder,tranforms = None,inv_normalize = None):
    model = torch.load('./model.h5')
    image = torch.from_numpy(np.expand_dims(image,axis = 0))
    model.eval()
    data = Variable(image)
    if(transforms):
        data = transforms(data)
    data = data.type(torch.FloatTensor).to(device)
    sm = nn.Softmax(dim = 1)
    output = model(data)
    output = sm(output)
    _, preds = torch.max(output, 1)
    img_plot(image,inv_normalize)
    prediction_bar(output,encoder)
