from speaksee.data import ImageField, TextField, RawField
from speaksee.data.pipeline import EncodeCNN
from speaksee.data.dataset import COCO_VQA
from torchvision.models import vgg16
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import torch
from torch import nn, optim
from speaksee.data import DataLoader
from speaksee.models import SAN

from tqdm import tqdm
import json
import requests
import argparse


def parse_arguments():
    """
    Parse arguments

    return: args
    """

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="/nas/softechict-nas-1/rdicarlo/COCO_VQA",
                        help='Dataset root path', metavar='')
    parser.add_argument('--vocab', type=str, default=None,
                        help='Dictionary of words to answer questions', metavar='')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Model mini-batch size', metavar='')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode: train | validation', metavar='')
    parser.add_argument('--save_model', type=str, default="/nas/softechict-nas-1/rdicarlo/vqa_checkpoint.pth",
                        help='Save model checkpoint', metavar='')
    parser.add_argument('--load_model', type=str, default="/nas/softechict-nas-1/rdicarlo/vqa_checkpoint.pth",
                        help='Load pre-trained model', metavar='')

    return parser.parse_args()


def validation(model, dataloader_val, criterion):
    """
    Validation loop

    Validation loop
    model: Stack attention model
    dataloader_val: data for validation
    criterion: loss

    return: None
    """

    for e in range(50):
        model.eval()
        running_loss = .0
        running_accuracy = .0
        with tqdm(desc='Epoch %d - val' % e, unit='it', total=len(dataloader_val)) as pbar:
            for it, (answ, images, questions) in enumerate(dataloader_val):
                images, questions, answ = images.to(device), questions.to(device), answ.to(device)

                out = model(images, questions)
                loss = criterion(out, answ)

                running_accuracy += float(torch.sum(torch.max(out, 1)[1] == answ).item()) / out.shape[0]
                running_loss += loss.item()

                pbar.set_postfix(loss=running_loss / (it + 1), acc=running_accuracy / (it + 1))
                pbar.update()


def train(model, dataloader_train, optimizer, criterion):
    """
    Train loop

    model: Stack attention model
    dataloader_train: Data for training
    optimizer: Ottimizzatore parametri
    criterion: Loss

    return: None
    """

    for e in range(50):
        model.train()
        running_loss = .0
        running_accuracy = .0
        with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader_train)) as pbar:
            for it, (answ, images, questions) in enumerate(dataloader_train):
                images, questions, answ = images.to(device), questions.to(device), answ.to(device)
                out = model(images, questions)
                optimizer.zero_grad()

                loss = criterion(out, answ)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_accuracy += float(torch.sum(torch.max(out, 1)[1] == answ).item()) / out.shape[0]

                pbar.set_postfix(loss=running_loss / (it + 1), acc=running_accuracy / (it + 1))
                pbar.update()

                # Serialize model
                if it % 1000 == 0:
                    torch.save({
                        'epoch': e,
                        'train_loss': running_loss / (it + 1),
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, args.save_model)


def main():
    """
    Main function

    return: None
    """

    # Preprocess with some fancy cnn and transformation
    cnn = vgg16(pretrained=True).to(device)
    cnn = nn.Sequential(*list(cnn.children())[:-1])

    transforms = Compose([
        Resize((448, 448)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    prepro_pipeline = EncodeCNN(cnn, transforms)
    image_field = ImageField(preprocessing=prepro_pipeline)

    # Download vocab if not exist
    if args.vocab is None:
        url = 'https://www.dropbox.com/s/6zd1cpxikf86lib/vocab.json?dl=1'
        r = requests.get(url)
        args.vocab = 'vocab.json'
        with open(args.vocab, 'wb') as f:
            f.write(r.content)

    with open(args.vocab) as f:
        vocab = json.load(f)

    # Pipeline for text
    questions = TextField(eos_token='<eos>', lower=True, tokenize='spacy', remove_punctuation=True)
    answers = RawField()

    dataset = COCO_VQA(image_field, questions, answers, vocab, args.dataset, args.dataset + "/annotations")
    train_dataset, val_dataset = dataset.splits
    questions.build_vocab(train_dataset, val_dataset, min_freq=5)

    # Load Data
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Load model, optimizer, loss function
    model = SAN(len(questions.vocab), len(vocab), num_attention=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-04)
    criterion = nn.CrossEntropyLoss().to(device)

    # Load pretrained model
    if args.load_model != '':
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint['state_dict'])

    if args.mode == 'train':
        train(model, dataloader_train, optimizer, criterion)
    elif args.mode == 'validation':
        validation(model, dataloader_val, criterion)


if __name__ == '__main__':
    # Entry point
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()
