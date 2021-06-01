import torch
from SSD_Pure import *
from datasets import *
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = '/home/vuong/PycharmProjects/hand-predict/hand-sign/data_ver2/data'  # folder with data files
checkpoint = "/home/vuong/PycharmProjects/hand-predict/checkpoint/checkpoint_ssd300.pth_SSD_pure.tar"

checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model.eval()

test_dataset = DataLoader(data_folder, split='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle= True,collate_fn=test_dataset.collate_fn, num_workers=2)
def mAP_per_class(model, test_loader):
    with torch.no_grad():
        det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties = [], [], [], [] ,[], []
        for i, (images, boxes, labels, _) in enumerate(test_loader):
            _ = [torch.Tensor([int(_[0][0])]).to(device)]
            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            det_box, det_label, det_score = model.detect_objects(predicted_locs, predicted_scores, min_score=0.1,
                                                                 max_overlap=0.5, top_k=100)
            det_boxes.extend(det_box)
            det_scores.extend(det_score)
            det_labels.extend(det_label)
            true_boxes.extend(boxes)
            true_labels.extend(labels)

            true_difficulties.extend(_)

    return det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties
det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties = mAP_per_class(model, test_loader)
p,r,k,n = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

def show_figue(lr_recall, lr_precision, i ):
    lr_recall = lr_recall[i][50:-20]
    lr_ppltrecision = lr_precision[i][50:-20]
    pyplot.plot(lr_recall.to('cpu'), lr_precision.to('cpu'), marker='.', label = rev_label_map[i+1])
    # axis labels
    pyplot.xlabel('Recall')
    pyplot.ylabel('Precision')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()
