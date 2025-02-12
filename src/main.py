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
  print('Images length:', len(images))
  
  # Split images into training and validation sets (80/20 split)
  train_size = int(0.8 * len(images))
  validate_size = len(images) - train_size
  train_images, validate_images = torch.utils.data.random_split(images, [train_size, validate_size])
  print('Training set size:', len(train_images))
  print('Validation set size:', len(validate_images))
  optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.001)

  train_images_pbar = tqdm(train_images)
  validate_images_pbar = tqdm(validate_images)
  total_train_loss = 0
  total_validate_loss = 0

  for i, (image, label) in enumerate(train_images_pbar):
    optimizer.zero_grad()

    image = resize_image(image)
    patches = segment_and_unroll(image)
    classification_probabilities = encoder_model(patches)
    train_loss = loss_function(classification_probabilities, label)
    total_train_loss += train_loss.item()
    average_train_loss = total_train_loss/(i+1)
    
    train_loss.backward()
    optimizer.step()
    train_images_pbar.set_postfix({
      'loss': f'{train_loss.item():.4f}',
      'average_loss': f'{average_train_loss:.4f}'
    })
  
    if i % 10_000 == 1:
      print('average training loss:', average_train_loss)
  
  for i, (image, label) in enumerate(validate_images_pbar):
    image = resize_image(image)
    patches = segment_and_unroll(image)
    classification_probabilities = encoder_model(patches)
    validate_loss = loss_function(classification_probabilities, label)
    total_validate_loss += validate_loss.item()
    average_validate_loss = total_validate_loss/(i+1)
    
    # Calculate accuracy
    predicted_label = torch.argmax(classification_probabilities)
    correct = (predicted_label == label).item()
    total_validate_accuracy = total_validate_accuracy + correct if i > 0 else correct
    average_validate_accuracy = total_validate_accuracy/(i+1)
    
    validate_images_pbar.set_postfix({
      'loss': f'{validate_loss.item():.4f}',
      'average_loss': f'{average_validate_loss:.4f}',
      'average_accuracy': f'{average_validate_accuracy:.1%}'
    })
  print('average validation loss:', average_validate_loss)

images = import_images()
encoder_model = Encoder()

train_encoder(encoder_model, images)
