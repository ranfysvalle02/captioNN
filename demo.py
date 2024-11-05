import os
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor
from torch import nn
from torch.optim import Adam
from collections import defaultdict
import string
import torchvision.models as models
from torchvision.models import ResNet50_Weights  # Import the weights enum

# Constants
NUM_SAMPLES = 1000
SHAPES = ['square', 'triangle']
UNSEEN_SHAPES = ['circle']  # For demonstration purposes
ALL_SHAPES = SHAPES + UNSEEN_SHAPES
COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'orange': (255, 165, 0),
    'purple': (128, 0, 128)
}
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_EPOCHS = 10
MAX_CAPTION_LENGTH = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to generate shape images
def generate_shape_image(shape, color, size=IMAGE_SIZE):
    image = np.zeros(size + (3,), dtype=np.uint8)
    cv2_color = COLORS[color]

    if shape == 'circle':
        center = (np.random.randint(50, size[0]-50), np.random.randint(50, size[1]-50))
        radius = np.random.randint(20, 50)
        cv2.circle(image, center, radius, cv2_color, thickness=-1)
    elif shape == 'square':
        top_left = (np.random.randint(0, size[0]-100), np.random.randint(0, size[1]-100))
        side_length = np.random.randint(50, 100)
        bottom_right = (top_left[0] + side_length, top_left[1] + side_length)
        cv2.rectangle(image, top_left, bottom_right, cv2_color, thickness=-1)
    elif shape == 'triangle':
        pt1 = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
        pt2 = (pt1[0] + np.random.randint(-50, 50), pt1[1] + np.random.randint(50, 100))
        pt3 = (pt1[0] + np.random.randint(-50, 50), pt1[1] + np.random.randint(-50, 50))
        points = np.array([pt1, pt2, pt3])
        cv2.fillPoly(image, [points], cv2_color)
    
    return image

# Function to create captions based on color and shape
def create_caption(color, shape):
    return f"A {color} {shape}."

# Vocabulary class
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = defaultdict(int)
        self.idx = 0
        self.add_word("<PAD>")
        self.add_word("<SOS>")
        self.add_word("<EOS>")
        self.add_word("<UNK>")

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocabulary(self, captions):
        for caption in captions:
            tokens = self.tokenize(caption)
            for token in tokens:
                self.word_freq[token] += 1
                if self.word_freq[token] == 1:
                    self.add_word(token)

    def tokenize(self, sentence):
        sentence = sentence.lower()
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        return sentence.split()

    def numericalize(self, caption):
        tokens = self.tokenize(caption)
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

    def decode(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, "<UNK>")
            if word == "<EOS>":
                break
            if word not in ["<PAD>", "<SOS>"]:
                words.append(word)
        return ' '.join(words)

# Custom Dataset
class ShapeCaptionDataset(Dataset):
    def __init__(self, num_samples, shapes, colors, transform=None, include_unseen=False):
        self.images = []
        self.captions = []
        self.shapes = shapes.copy()
        if include_unseen:
            self.shapes += UNSEEN_SHAPES
        self.colors = list(colors.keys())
        self.transform = transform

        for _ in range(num_samples):
            shape = np.random.choice(self.shapes)
            color = np.random.choice(self.colors)
            image = generate_shape_image(shape, color)
            caption = create_caption(color, shape)  # Renamed function
            self.images.append(image)
            self.captions.append(caption)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = self.captions[idx]
        if self.transform:
            image = self.transform(image)
        return image, caption

# Initialize Vocabulary
caption_vocab = Vocabulary()

# Function to build vocabulary
def build_vocabulary(dataset, vocab):
    captions = dataset.captions
    vocab.build_vocabulary(captions)

# Collate function for DataLoader
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions = [torch.tensor(caption_vocab.numericalize(cap) + [caption_vocab.word2idx["<EOS>"]]) for cap in captions]
    captions_padded = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=caption_vocab.word2idx["<PAD>"])
    return images, captions_padded

# CNN Encoder
class CNNEncoder(nn.Module):
    def __init__(self, encoded_image_size=8):
        super(CNNEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # Updated model loading with weights enum
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, H/32, W/32)
        features = self.adaptive_pool(features)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        features = features.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return features

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune the layers4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

# LSTM Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.init_hidden = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU()
        )
        self.init_cell = nn.Sequential(
            nn.Linear(2048, hidden_dim),
            nn.ReLU()
        )

    def forward(self, captions, features, lengths):
        lengths = lengths.cpu().type(torch.int64)  # Ensure lengths are on CPU and int64
        embeddings = self.embed(captions)  # (batch_size, seq_len, embed_dim)
        # Initialize hidden and cell states from image features
        h = self.init_hidden(features)  # (batch_size, hidden_dim)
        c = self.init_cell(features)    # (batch_size, hidden_dim)
        h = h.unsqueeze(0)  # (1, batch_size, hidden_dim)
        c = c.unsqueeze(0)  # (1, batch_size, hidden_dim)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed, (h, c))
        outputs = self.linear(outputs.data)  # (sum(lengths), vocab_size)
        return outputs

# Combined Model
class CNN_RNN(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, hidden_dim, learning_rate=1e-3):
        super(CNN_RNN, self).__init__()
        self.save_hyperparameters()
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(embed_dim, hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=caption_vocab.word2idx["<PAD>"])
        self.learning_rate = learning_rate

    def forward(self, images, captions, lengths):
        features = self.encoder(images)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        features = features.mean(dim=[1, 2])  # (batch_size, 2048)
        outputs = self.decoder(captions, features, lengths)  # (sum(lengths), vocab_size)
        return outputs

    def training_step(self, batch, batch_idx):
        images, captions = batch
        captions_input = captions[:, :-1]
        captions_target = captions[:, 1:]
        lengths = (captions_input != caption_vocab.word2idx["<PAD>"]).sum(dim=1)
        outputs = self.forward(images, captions_input, lengths)  # (sum(lengths), vocab_size)
        loss = self.criterion(outputs, captions_target.reshape(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch
        captions_input = captions[:, :-1]
        captions_target = captions[:, 1:]
        lengths = (captions_input != caption_vocab.word2idx["<PAD>"]).sum(dim=1)
        outputs = self.forward(images, captions_input, lengths)  # (sum(lengths), vocab_size)
        loss = self.criterion(outputs, captions_target.reshape(-1))
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Function to generate caption using the trained model
def generate_caption(model, image, vocab, max_length=MAX_CAPTION_LENGTH):
    model.eval()
    transform = transforms.Compose([ToTensor()])
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encoder(image)  # (1, encoded_image_size, encoded_image_size, 2048)
        features = features.mean(dim=[1, 2])  # (1, 2048)
    # Initialize hidden and cell states
    h = model.decoder.init_hidden(features)  # (batch_size, hidden_dim)
    c = model.decoder.init_cell(features)    # (batch_size, hidden_dim)
    h = h.unsqueeze(0)  # (1, batch_size, hidden_dim)
    c = c.unsqueeze(0)  # (1, batch_size, hidden_dim)
    
    caption = [vocab.word2idx["<SOS>"]]
    input_caption = torch.tensor(caption).unsqueeze(0).to(DEVICE)  # (1, 1)
    for _ in range(max_length):
        embeddings = model.decoder.embed(input_caption)  # (1, 1, embed_dim)
        outputs, (h, c) = model.decoder.lstm(embeddings, (h, c))  # outputs: (1,1,hidden_dim)
        logits = model.decoder.linear(outputs.squeeze(1))  # (1, vocab_size)
        predicted = logits.argmax(1).item()
        if predicted == vocab.word2idx["<EOS>"]:
            break
        caption.append(predicted)
        input_caption = torch.tensor([predicted]).unsqueeze(0).to(DEVICE)  # (1,1)
    return vocab.decode(caption[1:])

def main():
    # Create Dataset
    dataset = ShapeCaptionDataset(NUM_SAMPLES, SHAPES, COLORS, transform=transforms.Compose([ToTensor()]))
    build_vocabulary(dataset, caption_vocab)

    # Split Dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data Loaders with num_workers=0 to prevent multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Initialize Model
    model = CNN_RNN(vocab_size=len(caption_vocab.word2idx), embed_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)

    # Initialize Trainer with Updated Arguments
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='auto',
        devices='auto',
        enable_progress_bar=True,
        log_every_n_steps=10
    )

    # Train the Model
    trainer.fit(model, train_loader, val_loader)

    # Save the trained model
    torch.save(model.state_dict(), "cnn_rnn_caption_model.pth")

    # Load the model for inference
    model = CNN_RNN(vocab_size=len(caption_vocab.word2idx), embed_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load("cnn_rnn_caption_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    # Demonstration
    print("=== Caption Generation Demo ===\n")

    # Generate a seen shape (e.g., triangle)
    seen_shape = 'triangle'
    seen_color = 'blue'
    seen_image = generate_shape_image(seen_shape, seen_color)
    seen_caption = generate_caption(model, seen_image, caption_vocab)
    print(f"Generated Caption for seen shape ({seen_shape}, {seen_color}): {seen_caption}")

    # Generate an unseen shape (e.g., circle)
    unseen_shape = 'circle'
    unseen_color = 'red'
    unseen_image = generate_shape_image(unseen_shape, unseen_color)
    unseen_caption = generate_caption(model, unseen_image, caption_vocab)
    print(f"Generated Caption for unseen shape ({unseen_shape}, {unseen_color}): {unseen_caption}")

if __name__ == "__main__":
    main()


"""
=== Caption Generation Demo ===

Generated Caption for seen shape (triangle, blue): 
Generated Caption for unseen shape (circle, red): 
"""
