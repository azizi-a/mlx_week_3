import torch
import torchvision
from tqdm import tqdm
from encoder import Encoder

encoder = Encoder()

def import_images():
    mnist_data = torchvision.datasets.MNIST(root='./data', 
                              train=True, 
                              download=True, 
                              transform=torchvision.transforms.ToTensor())
    return mnist_data

def resize_image(image):
  return torch.nn.functional.interpolate(image.unsqueeze(0), size=(56, 56), mode='bilinear').squeeze(0)

def segment_and_unroll(image):
  # Take patches of 14x14 pixels each and unroll them into vectors
  patches = image.flatten().unfold(0, 14 * 14, 14 * 14)
  return patches

def loss_function(predictions, label):
  expected_output = torch.zeros(10)
  expected_output[label] = 1
  return torch.nn.functional.cross_entropy(
    # Get the index of the highest probability prediction
    predictions,
    expected_output
  )

def train_encoder(encoder_model, images):
  print('images length:', len(images))
  optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.001)

  train_images = images
  train_images_pbar = tqdm(train_images)

  total_loss = 0

  for i, (image, label) in enumerate(train_images_pbar):
    optimizer.zero_grad()

    image = resize_image(image)
    patches = segment_and_unroll(image)
    classification_probabilities = encoder_model(patches)
    loss = loss_function(classification_probabilities, label)
    total_loss += loss.item()
    average_loss = total_loss/(i+1)
    
    loss.backward()
    optimizer.step()
    train_images_pbar.set_postfix({
      'loss': loss.item(),
      'average_loss': average_loss
    })
  
    if i % 1000 == 0:
      print('average loss:', average_loss)

images = import_images()
encoder_model = Encoder()

train_encoder(encoder_model, images)
