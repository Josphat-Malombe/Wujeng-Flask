import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io



FOOD101_CLASSES = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', 
                   'bibimbap', 'bread_pudding', 'breakfast_burrito', 'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad',
                     'carrot_cake', 'ceviche', 'cheese_plate', 'cheesecake', 'chicken_curry', 'chicken_quesadilla', 
                     'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder', 'club_sandwich', 
                     'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 'deviled_eggs', 'donuts', 'dumplings', 'edamame',
                       'eggs_benedict', 'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 
                       'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 'garlic_bread', 
                       'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 
                       'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 
                       'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette',
                         'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza',
                    'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
                      'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 
                      'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
                      ]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, width, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, width * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, width * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(width * self.expansion)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)



class ResNet50(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

       
        self.bn_final = nn.BatchNorm1d(512 * Bottleneck.expansion * 2)
        self.fc = nn.Linear(512 * Bottleneck.expansion * 2, num_classes)

    def _make_layer(self, block, width, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, width, stride))
        self.in_channels = width * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, width))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).flatten(1)

        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.bn_final(x)
        return self.fc(x)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
PREPROCESS = None

MEAN = [0.5463075637817383, 0.4448153078556061, 0.3447556495666504]
STD  = [0.2698375880718231, 0.2727035880088806, 0.2779143750667572]
IMG_SIZE = 384




def load_model(model_path):
    global MODEL, PREPROCESS

    MODEL = ResNet50(num_classes=101)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    MODEL.load_state_dict(checkpoint["model"])
    MODEL.to(DEVICE)
    MODEL.eval()

    PREPROCESS = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


    print(f"[INFO] Model loaded on {DEVICE}")


def predict_from_bytes(image_bytes: bytes) -> dict:
   
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
   
    img = PREPROCESS(img).unsqueeze(0).to(DEVICE) 
    
  
    with torch.no_grad():
        logits = MODEL(img)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    
  
    top5_prob, top5_idx = torch.topk(probs, 5)
    print("Top5 probs:", top5_prob.cpu().numpy())
    print("Top5 classes:", [FOOD101_CLASSES[i] for i in top5_idx[0].cpu().numpy()])
    
    
    cls = FOOD101_CLASSES[idx.item()]
    
    return {
        "result": "hotdog" if cls == "hot_dog" else "not hotdog",
        "confidence": round(conf.item(), 4),
        "raw_class": cls.replace("_", " ").title(),
        "class_index": idx.item(),
    }


