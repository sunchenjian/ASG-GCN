import torch
import numpy as np

def load_data(label_dir, feature_dir):

    labels = np.load(label_dir, allow_pickle=True)

    y = labels
    y = y.astype(np.long)
    raw_features = np.load(feature_dir, allow_pickle=True)
    raw_features = raw_features.astype(float)

    return raw_features, y

if __name__ == '__main__':
    # model = torch.load("model_saved.pt")
    # print(model)
    # del model.classifier
    # print(model)
    # torch.save(model, r"C:\Users\sun\Desktop\models\model_5\emo_model.pt")
    device = torch.device("cuda")

    label_dir = r'C:\Users\sun\Desktop\models\features\5s_random\all_label_daic.npy'
    feature_dir = r'C:\Users\sun\Desktop\models\features\5s_random\all_data_daic.npy'

    raw_features, y = load_data(label_dir, feature_dir)
    model = torch.load(r"C:\Users\sun\Desktop\models\model_5\emo_model.pt").to(device)
    print(model)
    y = model(torch.Tensor(raw_features).to(device))
    print(y.shape)


