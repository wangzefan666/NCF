import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1.
	return 0.


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item) + 1  # get pos item's rank, it should be added by one
		return np.reciprocal(np.log2(index+1.))  # np.reciprocal, get element-wise reciprocal result
	return 0.


def metrics(model, test_loader, top_k):
	HR, NDCG = [], []
	# a batch is a testing user
	for user, item, label in test_loader:
		user = user.cuda()
		item = item.cuda()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)  # return (num, indices)
		recommends = torch.take(item, indices).cpu().numpy().tolist()  # topk items' ids

		gt_item = item[0].item()  # the only pos item
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG)
