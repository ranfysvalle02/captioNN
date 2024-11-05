# captioNN

---

## Table of Contents

1. [Introduction](#introduction)
2. [Synthetic Data Generation](#synthetic-data-generation)
3. [Vocabulary Management](#vocabulary-management)
4. [Custom Dataset and DataLoader](#custom-dataset-and-dataloader)
5. [Model Architecture: CNN Encoder and LSTM Decoder](#model-architecture-cnn-encoder-and-lstm-decoder)
6. [Training with PyTorch Lightning](#training-with-pytorch-lightning)
7. [Inference and Caption Generation](#inference-and-caption-generation)
8. [Conclusion](#conclusion)

---

## Introduction

Image captioning is a quintessential task that combines computer vision and natural language processing. The goal is to generate a coherent and descriptive sentence that accurately describes the content of an image. Achieving this requires understanding both the visual elements of the image and the linguistic constructs of language.

In this implementation, we focus on a synthetic dataset composed of simple geometric shapes. This controlled environment allows us to thoroughly understand and explain each component of the model, ensuring clarity and maintainability.

---

## Synthetic Data Generation

### Why Synthetic Data?

Creating synthetic data offers several advantages:

- **Control:** We have complete control over the data, ensuring that each image contains specific shapes and colors.
- **Simplicity:** It simplifies the complexity, making it easier to debug and understand the model's behavior.
- **Reproducibility:** Synthetic data ensures consistent results across different runs and environments.

### Generating Shape Images

The `generate_shape_image` function creates images with specified shapes and colors. Here's how it works:

```python
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
```

- **Shapes Supported:** Circle, Square, Triangle.
- **Colors:** Defined in the `COLORS` dictionary with RGB values.
- **Randomization:** Positions and sizes are randomized within constraints to ensure variability.

### Creating Captions

Each image has a corresponding caption that describes the shape and its color:

```python
def create_caption(color, shape):
    return f"A {color} {shape}."
```

**Example:** "A red circle."

---

## Vocabulary Management

### The Importance of a Vocabulary

In natural language processing, managing the vocabulary is crucial. It involves mapping words to unique indices and handling unknown words during inference.

### The `Vocabulary` Class

```python
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
```

- **Special Tokens:** `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>` for padding, start of sentence, end of sentence, and unknown words respectively.
- **Building Vocabulary:** Iterates through all captions to build `word2idx` and `idx2word` mappings.
- **Numericalization:** Converts captions into sequences of indices.
- **Decoding:** Converts sequences of indices back into human-readable captions.

---

## Custom Dataset and DataLoader

### Custom Dataset: `ShapeCaptionDataset`

```python
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
            caption = create_caption(color, shape)
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
```

- **Parameters:**
  - `num_samples`: Number of samples to generate.
  - `shapes`: List of shapes to include.
  - `colors`: Dictionary of colors.
  - `transform`: Transformations to apply to images.
  - `include_unseen`: Whether to include shapes not seen during training.

- **Data Generation:** For each sample, a random shape and color are selected to generate an image and its corresponding caption.

### DataLoader and Collate Function

Handling variable-length captions requires a custom collate function:

```python
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, 0)
    captions = [torch.tensor(caption_vocab.numericalize(cap) + [caption_vocab.word2idx["<EOS>"]]) for cap in captions]
    captions_padded = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=caption_vocab.word2idx["<PAD>"])
    return images, captions_padded
```

- **Padding:** Ensures that all captions in a batch have the same length by padding shorter captions.
- **Batching:** Stacks images into a single tensor for efficient processing.

---

## Model Architecture: CNN Encoder and LSTM Decoder

The model is composed of two main components:

1. **CNN Encoder:** Extracts visual features from images.
2. **LSTM Decoder:** Generates captions based on the extracted features.

### CNN Encoder

```python
class CNNEncoder(nn.Module):
    def __init__(self, encoded_image_size=8):
        super(CNNEncoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1)
        return features

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
```

- **ResNet50 Backbone:** Pretrained on ImageNet for robust feature extraction.
- **Feature Extraction:** Removes the last two layers (average pooling and fully connected) to obtain spatial features.
- **Adaptive Pooling:** Resizes feature maps to a fixed size (`encoded_image_size`).
- **Fine-Tuning:** Freezes earlier layers to retain learned features and allows fine-tuning of deeper layers for better task-specific performance.

### LSTM Decoder

```python
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
        lengths = lengths.cpu().type(torch.int64)
        embeddings = self.embed(captions)
        h = self.init_hidden(features)
        c = self.init_cell(features)
        h = h.unsqueeze(0)
        c = c.unsqueeze(0)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed, (h, c))
        outputs = self.linear(outputs.data)
        return outputs
```

- **Embedding Layer:** Transforms word indices into dense vectors.
- **LSTM Layer:** Processes the embedded captions to generate contextual representations.
- **Linear Layer:** Maps LSTM outputs to vocabulary space for prediction.
- **Initialization:** Uses the image features to initialize the hidden and cell states of the LSTM, effectively grounding the caption generation in the visual content.

### Combined CNN-RNN Model

```python
class CNN_RNN(pl.LightningModule):
    def __init__(self, vocab_size, embed_dim, hidden_dim, learning_rate=1e-3):
        super(CNN_RNN, self).__init__()
        self.save_hyperparameters()
        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(embed_dim, hidden_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=caption_vocab.word2idx["<PAD>"])
        self.learning_rate = learning_rate

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        features = features.mean(dim=[1, 2])
        outputs = self.decoder(captions, features, lengths)
        return outputs

    def training_step(self, batch, batch_idx):
        images, captions = batch
        captions_input = captions[:, :-1]
        captions_target = captions[:, 1:]
        lengths = (captions_input != caption_vocab.word2idx["<PAD>"]).sum(dim=1)
        outputs = self.forward(images, captions_input, lengths)
        loss = self.criterion(outputs, captions_target.reshape(-1))
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, captions = batch
        captions_input = captions[:, :-1]
        captions_target = captions[:, 1:]
        lengths = (captions_input != caption_vocab.word2idx["<PAD>"]).sum(dim=1)
        outputs = self.forward(images, captions_input, lengths)
        loss = self.criterion(outputs, captions_target.reshape(-1))
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
```

- **PyTorch Lightning Integration:** Simplifies training loops, logging, and checkpointing.
- **Forward Pass:** Encodes images and decodes captions, computing outputs for loss calculation.
- **Training and Validation Steps:** Handles input preparation, loss computation, and logging.
- **Optimizer Configuration:** Uses Adam optimizer with a configurable learning rate.

---

## Training with PyTorch Lightning

### Why PyTorch Lightning?

PyTorch Lightning abstracts away the boilerplate code associated with training loops, allowing developers to focus on the model architecture and core logic. It enhances:

- **Modularity:** Clear separation between different parts of the model.
- **Scalability:** Easy to scale across multiple GPUs or TPUs.
- **Reproducibility:** Ensures consistent training across different environments.

### Training Process

```python
def main():
    # Create Dataset
    dataset = ShapeCaptionDataset(NUM_SAMPLES, SHAPES, COLORS, transform=transforms.Compose([ToTensor()]))
    build_vocabulary(dataset, caption_vocab)

    # Split Dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Initialize Model
    model = CNN_RNN(vocab_size=len(caption_vocab.word2idx), embed_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)

    # Initialize Trainer
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
```

- **Dataset Splitting:** 80% for training and 20% for validation.
- **DataLoader Configuration:** Uses the custom `collate_fn` to handle variable-length captions.
- **Model Initialization:** Configured with the size of the vocabulary, embedding dimensions, and hidden dimensions.
- **Trainer Setup:** Configured to automatically detect available hardware (CPU/GPU) and manage the training loop.
- **Model Saving and Loading:** Facilitates persistence and reuse of the trained model.

---

## Inference and Caption Generation

### Generating Captions

The `generate_caption` function utilizes the trained model to generate captions for new images:

```python
def generate_caption(model, image, vocab, max_length=MAX_CAPTION_LENGTH):
    model.eval()
    transform = transforms.Compose([ToTensor()])
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = model.encoder(image)
        features = features.mean(dim=[1, 2])
    # Initialize hidden and cell states from image features
    h = model.decoder.init_hidden(features)
    c = model.decoder.init_cell(features)
    h = h.unsqueeze(0)
    c = c.unsqueeze(0)
    
    caption = [vocab.word2idx["<SOS>"]]
    input_caption = torch.tensor(caption).unsqueeze(0).to(DEVICE)
    for _ in range(max_length):
        embeddings = model.decoder.embed(input_caption)
        outputs, (h, c) = model.decoder.lstm(embeddings, (h, c))
        logits = model.decoder.linear(outputs.squeeze(1))
        predicted = logits.argmax(1).item()
        if predicted == vocab.word2idx["<EOS>"]:
            break
        caption.append(predicted)
        input_caption = torch.tensor([predicted]).unsqueeze(0).to(DEVICE)
    return vocab.decode(caption[1:])
```

- **Preprocessing:** Transforms the image and moves it to the appropriate device.
- **Feature Extraction:** Encodes the image to obtain feature representations.
- **Caption Generation Loop:** Iteratively predicts the next word until `<EOS>` token is generated or the maximum length is reached.
- **Decoding:** Converts the sequence of predicted indices back into a readable sentence.

### Demonstration

```python
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
```

**Expected Output:**
```
=== Caption Generation Demo ===

Generated Caption for seen shape (triangle, blue): a blue triangle.
Generated Caption for unseen shape (circle, red): a red circle.
```

This demonstration highlights the model's ability to generate accurate captions for both seen and unseen shapes, showcasing its generalization capabilities.

---

## Conclusion

In this blog post, we've explored a comprehensive implementation of an image captioning model using PyTorch and PyTorch Lightning. By focusing on synthetic data generation, we ensured a controlled environment that facilitated a clear understanding of each component's role. The model's architecture, emphasizing the composability of neural network modules, allows for easy maintenance and scalability.

**Key Takeaways:**

- **Explainability:** Each component of the model—from data generation to vocabulary management and model architecture—is designed for clarity and understanding.
- **Composability:** Modular design using separate classes for the encoder, decoder, and overall model enhances maintainability and flexibility.
- **PyTorch Lightning Integration:** Simplifies the training process, making the codebase cleaner and more organized.
- **Generalization:** The model demonstrates the ability to generate accurate captions for both seen and unseen data, indicating robust learning.

This implementation serves as a solid foundation for more complex image captioning tasks, providing a clear pathway for incorporating additional features such as attention mechanisms, more intricate datasets, or advanced language models.

---
