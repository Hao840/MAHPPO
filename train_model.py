import argparse
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from utils import setup_logger
from dataset import get_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/')
    parser.add_argument('--ckp', type=str, default='ckp/')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'vgg11', 'mobilenetv2'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    exp_name = f'model_train_{args.model}'
    os.makedirs(os.path.join('result', exp_name), exist_ok=True)
    logger = setup_logger(__name__, os.path.join('result', exp_name))

    if args.model == 'resnet18':
        net = torchvision.models.resnet18(num_classes=101).cuda()
    elif args.model == 'vgg11':
        net = torchvision.models.vgg11_bn(num_classes=101).cuda()
    elif args.model == 'mobilenetv2':
        net = torchvision.models.mobilenet_v2(num_classes=101).cuda()
    else:
        raise NotImplementedError

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [80, 120], 0.1)
    train_loader, test_loader = get_loader('caltech101', args.batch_size)

    for e in range(args.epochs):
        # train
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            inputs, labels = images.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        logger.info(f'Epoch {e}/{args.epochs}, Loss: {loss.item():.4f}')

        # test
        net.eval()
        total_correct = 0
        losses = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                inputs, labels = images.cuda(), labels.cuda()
                outputs = net(inputs)
                losses += F.cross_entropy(outputs, labels).item()
                pred = outputs.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)
                                         ).sum().item()
        loss = round(losses / len(test_loader.dataset), 4)
        acc = round(total_correct / len(test_loader.dataset), 4)
        logger.info(f'Test Avg. Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # save model
        torch.save(net.state_dict(), os.path.join('result', exp_name, 'ckp.pt'))
